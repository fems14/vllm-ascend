# Standard
from typing import List, Optional, Tuple
import abc
import time

# Third Party
import torch

# First Party
from vllm_ascend.distributed.mooncake_store import Mooncakestore


class NPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def npu_h2d(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Store the data in the memory object into a NPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into NPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def npu_d2h(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Load the data from a NPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            NPU.
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
    The NPU KV cache should be a nested tuple of K and V tensors.
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
        store: Mooncakestore,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers*2, dtype=torch.int64, device="cpu"
        ) 
        self.page_buffer_size = 0
        self.store=store

    def npu_h2d(self, memory_tensor: torch.Tensor, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching


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
        self.multi_layer_kv_transfer_py(
            memory_tensor, 
            kvcaches, 
            slot_mapping[start:end],
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
             should NEVER overlap with the prefix caching.

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
        
        self.multi_layer_kv_transfer_py(
            memory_tensor, 
            kvcaches, 
            slot_mapping[start:end],
            direction=True
        )
    
    def npu_h2d_layer(self, memory_tensor: torch.Tensor, start: int, end: int, kvcache:List[torch.Tensor], **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching


        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_tensor is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.layer_kv_transfer_py(
            memory_tensor,
            kvcache,
            slot_mapping[start:end],
            direction=False
        )
    
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

        kvcaches: torch.Tensor = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        
        self.layer_kv_transfer_py(
            memory_tensor,
            kvcaches,
            slot_mapping[start:end],
            direction=True
        )

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])
    
    def get_layer_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, 1, num_tokens, self.hidden_dim_size])
    
    def layer_kv_transfer_py(
        self,
        key_value: torch.Tensor,          # [2, num_layers, num_tokens, scalars_per_token]
        kv_caches: List[torch.Tensor],          # [2,block_num,block_size,num_heads, head_size]
        slot_mapping: torch.Tensor,       # [num_tokens]
        direction: bool, 
    ):
        assert key_value.dim() == 4, "key_value must be 4D [2, num_layers, num_tokens, scalars_per_token]"
        num_tokens = slot_mapping.shape[0]
        scalars_per_token = key_value.shape[-1]

        num_heads = kv_caches[0][0].shape[-2]
        head_size = kv_caches[0][0].shape[-1]

        assert scalars_per_token == num_heads * head_size, "hidden_dim 必须等于 num_heads * head_size"


        flat_key_value = key_value.view(-1)
        flat_kv_ptrs =[ptr.view(-1) for ptr in kv_caches]  # layer_kv: [K_tensor, V_tensor]

        kv_total = num_tokens * scalars_per_token
        slot_idx = slot_mapping[0].item()
        if slot_idx < 0:
            raise
        for k_or_v in [0, 1]:  # 0=Key, 1=Values
            # Compute LMCache start offset
            kv_offset = k_or_v * kv_total 
            # Compute paged KV offset
            kv_cache_offset = slot_idx * scalars_per_token

            kv_ptr=flat_key_value[kv_offset : kv_offset+1].data_ptr()
            kv_cache_ptr=flat_kv_ptrs[k_or_v][kv_cache_offset : kv_cache_offset+1].data_ptr()

            if direction:  # d2h
                self.store.store.d2h(kv_cache_ptr, kv_ptr, scalars_per_token*flat_key_value.element_size()*num_tokens)
            else:          # h2d
                self.store.store.h2d(kv_cache_ptr, kv_ptr, scalars_per_token*flat_key_value.element_size()*num_tokens)

    def multi_layer_kv_transfer_py(
        self,
        key_value: torch.Tensor,          # [2, num_layers, num_tokens, scalars_per_token]
        key_value_ptrs: list[list[torch.Tensor]],  # [num_layers][2], tensor [page_buffer_size, 2, scalars_per_token/2]
        slot_mapping: torch.Tensor,       # [num_tokens]
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
        slot_idx = slot_mapping[0].item()
        if slot_idx < 0:
            raise

        for layer_id in range(num_layers):
            for k_or_v in [0, 1]:  # 0=Key, 1=Value
                # Compute LMCache start offset
                kv_offset = (
                    k_or_v * kv_total +
                    layer_id * num_tokens * scalars_per_token
                )
                # Compute paged KV offset
                kv_cache_offset = slot_idx * scalars_per_token

                kv_ptr=flat_key_value[kv_offset : kv_offset+1].data_ptr()
                kv_cache_ptr=flat_kv_ptrs[layer_id][k_or_v][kv_cache_offset : kv_cache_offset + 1].data_ptr()

                # Perform copy
                if direction:
                    self.store.store.d2h(kv_cache_ptr, kv_ptr, scalars_per_token*flat_key_value.element_size()*num_tokens)
                else:
                    self.store.store.h2d(kv_cache_ptr, kv_ptr, scalars_per_token*flat_key_value.element_size()*num_tokens)

class PagedMemNPUConnectorMLA(NPUConnectorInterface):
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
        store: Mooncakestore,
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
        self.store=store


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
        print("memory_tensor.shape:", memory_tensor.shape)
        self.multi_layer_kv_transfer_py_mla(
            memory_tensor, 
            kvcaches, 
            slot_mapping[start:end],
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
        print("memory_tensor.shape:", memory_tensor.shape)
        self.multi_layer_kv_transfer_py_mla(
            memory_tensor, 
            kvcaches, 
            slot_mapping[start:end],
            direction=True
        )
        torch.npu.synchronize()
        
    def multi_layer_kv_transfer_py_mla(
        self,
        key_value: torch.Tensor,          # [1, num_layers, num_tokens, scalars_per_token] [1,27,15,576] 
        key_value_ptrs: list[torch.Tensor],  # [num_layers][1], tensor [page_buffer_size, 1, scalars_per_token] [27][2] tensor [6286, 128, 1, 512] [6286, 128, 1, 64]
        slot_mapping: torch.Tensor,       # [num_tokens]
        direction: bool,                  # True: Paged->LMCache, False: LMCache->Paged
    ):
        # assert key_value.dim() == 4, "key_value must be 4D [2, num_layers, num_tokens, scalars_per_token]"
        num_layers = len(key_value_ptrs)
        num_tokens = slot_mapping.size(0)
        k_per_token = key_value_ptrs[0][0].size(3)
        v_per_token = key_value_ptrs[0][1].size(3)

        # Flattened view of key_value tensor
        flat_key_value = key_value.view(-1)
        
        # Flatten paged buffer once
        flat_kv_ptrs = [
            [ptr.view(-1) for ptr in layer_kv]  # layer_kv: [K_tensor, V_tensor]
            for layer_kv in key_value_ptrs
        ]  

        slot_idx = slot_mapping[0].item()
        if slot_idx < 0:
            raise  # skip invalid slots
    
        for layer_id in range(num_layers):
            for k_or_v in [0, 1]:
                if k_or_v == 0:
                    kv_offset = (
                        layer_id * num_tokens * k_per_token
                    )
                    kv_cache_offset = slot_idx * k_per_token
                    kv_ptr=flat_key_value[kv_offset : kv_offset+1].data_ptr()
                    kv_cache_ptr=flat_kv_ptrs[layer_id][k_or_v][kv_cache_offset : kv_cache_offset+1].data_ptr()
                    num_tokens_len = k_per_token*flat_key_value.element_size()*num_tokens
                elif k_or_v == 1:
                    kv_offset = (
                        num_layers * num_tokens * k_per_token +
                        layer_id * num_tokens * v_per_token
                    )
                    kv_cache_offset = slot_idx * v_per_token
                    kv_ptr=flat_key_value[kv_offset : kv_offset+1].data_ptr()
                    kv_cache_ptr=flat_kv_ptrs[layer_id][k_or_v][kv_cache_offset : kv_cache_offset+1].data_ptr()
                    num_tokens_len = v_per_token*flat_key_value.element_size()*num_tokens
            
                if direction:
                    self.store.store.d2h(kv_cache_ptr, kv_ptr, num_tokens_len)
                else:
                    self.store.store.h2d(kv_cache_ptr, kv_ptr, num_tokens_len)
    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([1, self.num_layers*num_tokens*self.hidden_dim_size])
