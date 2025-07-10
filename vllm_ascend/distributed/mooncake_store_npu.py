# Standard
from typing import List, Optional, Tuple
import abc

# Third Party
import torch

# First Party



def multi_layer_kv_transfer_py(
    key_value: torch.Tensor,          # [2, num_layers, num_tokens, scalars_per_token]
    key_value_ptrs: list[list[torch.Tensor]],  # [num_layers][2], tensor [page_buffer_size, 2, scalars_per_token/2]
    slot_mapping: torch.Tensor,       # [num_tokens]
    device: torch.device,
    page_buffer_size: int,
    direction: bool,                  # True: Paged->LMCache, False: LMCache->Paged
):
    assert key_value.dim() == 4, "key_value must be 4D [2, num_layers, num_tokens, scalars_per_token]"
    
    num_layers = key_value.size(1)
    num_tokens = slot_mapping.size(0)
    scalars_per_token = key_value.size(3)

    # Flattened view of key_value tensor
    flat_key_value = key_value.view(-1)

    # Flatten paged buffer once
    flat_kv_ptrs = [
        [ptr.view(-1) for ptr in layer_kv]  # layer_kv: [K_tensor, V_tensor]
        for layer_kv in key_value_ptrs
    ]

    kv_total = num_layers * num_tokens * scalars_per_token
    for token_id in range(num_tokens):
        slot_idx = slot_mapping[token_id].item()
        if slot_idx < 0:
            continue  # skip invalid slots

        for layer_id in range(num_layers):
            for k_or_v in [0, 1]:  # 0=Key, 1=Value
                # Compute LMCache start offset
                lmcache_offset = (
                    k_or_v * kv_total +
                    layer_id * num_tokens * scalars_per_token +
                    token_id * scalars_per_token
                )
                # Compute paged KV offset
                vllm_offset = slot_idx * scalars_per_token

                # Get slice views
                lmcache_slice = flat_key_value[lmcache_offset : lmcache_offset + scalars_per_token]
                paged_slice = flat_kv_ptrs[layer_id][k_or_v][vllm_offset : vllm_offset + scalars_per_token]

                # Perform copy
                if direction:  # Paged -> LMCache
                    lmcache_slice.copy_(paged_slice)
                else:          # LMCache -> Paged
                    paged_slice.copy_(lmcache_slice)


def layer_kv_transfer_py(
    key_value: torch.Tensor,          # [2, num_layers, num_tokens, scalars_per_token]
    kv_caches: torch.Tensor,          # [2,block_num,block_size,num_heads, head_size]
    slot_mapping: torch.Tensor,       # [num_tokens]
    device: torch.device,
    page_buffer_size: int,
    direction: bool,                  # True: Paged->LMCache, False: LMCache->Paged
):
    assert key_value.dim() == 4, "key_value must be 4D [2, num_layers, num_tokens, scalars_per_token]"
    num_tokens = slot_mapping.shape[0]
    hidden_dim = key_value.shape[-1]

    num_heads = kv_caches[0][0].shape[-2]
    head_size = kv_caches[0][0].shape[-1]
    block_size = kv_caches[0][0].shape[-3]

    assert hidden_dim == num_heads * head_size, "hidden_dim 必须等于 num_heads * head_size"

    k_tensor, v_tensor = kv_caches  # 分别是 shape: [num_blocks, block_size, num_heads, head_size]
    for token_idx in range(num_tokens):
        slot_idx = slot_mapping[token_idx].item()
        if slot_idx < 0:
            continue  # skip invalid slots
        if direction:
            key_value[0, 0, token_idx] = \
                k_tensor[(int)(slot_idx//block_size), (int)(slot_idx%block_size)].reshape(-1)
            key_value[1, 0, token_idx] = \
                v_tensor[(int)(slot_idx//block_size), (int)(slot_idx%block_size)].reshape(-1)
        else:
            k_tensor[(int)(slot_idx//block_size), (int)(slot_idx%block_size)] = \
                key_value[0, 0, token_idx].reshape(num_heads, head_size)
            v_tensor[(int)(slot_idx//block_size), (int)(slot_idx%block_size)] = \
                key_value[1, 0, token_idx].reshape(num_heads, head_size)

class NPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def npu_h2d(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def npu_d2h(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError


class PagedMemNPUConnector(NPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers*2, dtype=torch.int64, device="cpu"################  num_layers*2 num_layers
        ) 
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0



    def _initialize_pointers(self, kv_caches) -> torch.Tensor:
        # 展开 data_ptr（兼容 List[Tensor] 和 List[List[Tensor]]）
        if isinstance(kv_caches[0], list):
            # NPU 风格：每层有 [K, V]
            flat_ptrs = [t.data_ptr() for layer in kv_caches for t in layer]  # 2x num_layers
            num_layers = len(kv_caches)
        else:
            # GPU 风格：每层一个 Tensor，内部已堆叠
            flat_ptrs = [t.data_ptr() for t in kv_caches]
            num_layers = len(kv_caches)

        # 写入到 numpy 指针数组
        self.kv_cache_pointers.numpy()[:len(flat_ptrs)] = flat_ptrs

        device = kv_caches[0][0].device if isinstance(kv_caches[0], list) else kv_caches[0].device
        # assert device.type == "cuda", "The device should be CUDA."
        idx = device.index

        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                len(flat_ptrs), dtype=torch.int64, device=device
            )

        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers[:len(flat_ptrs)])

        # page buffer size 推测逻辑（从任意一个 Tensor 拿即可）
        sample_tensor = kv_caches[0][0] if isinstance(kv_caches[0], list) else kv_caches[0]
        self.page_buffer_size = sample_tensor.shape[1] * sample_tensor.shape[2]

        return self.kv_cache_pointers_on_gpu[idx]

    def npu_h2d(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        multi_layer_kv_transfer_py(
            memory_tensor, 
            kvcaches, 
            slot_mapping[start:end],
            kvcaches[0][0].device, 
            self.page_buffer_size,
            direction=False
        )
    def npu_h2d_layer(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: torch.Tensor = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        layer_kv_transfer_py(
            memory_tensor,
            kvcaches,
            slot_mapping[start:end],
            kvcaches.device,
            self.page_buffer_size,
            direction=False
        )
    def npu_d2h(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_tensor is not None
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        multi_layer_kv_transfer_py(
            memory_tensor, 
            kvcaches, 
            slot_mapping[start:end],
            kvcaches[0][0].device, 
            self.page_buffer_size,
            direction=True
        )
        torch.npu.synchronize()
    def npu_d2h_layer(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_tensor is not None
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        layer_kv_transfer_py(
            memory_tensor,
            kvcaches,
            slot_mapping[start:end],
            kvcaches.device,
            self.page_buffer_size,
            direction=True
        )
        torch.npu.synchronize()

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def get_shape_layer(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, 1, num_tokens, self.hidden_dim_size])

