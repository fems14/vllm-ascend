# Standard
from typing import Dict, Generator, List, Optional, Union
import asyncio
import multiprocessing
import time
import threading
import queue
from dataclasses import dataclass

# Third Party
import torch, torch_npu
from vllm.utils import cdiv, get_kv_cache_torch_dtype, round_down
from vllm.utils import logger
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm_ascend.distributed.config_data import MoonCakeEngineKey, MoonCakeEngineMetadata, ChunkedTokenDatabase, LayerMoonCakeEngineKey
from vllm_ascend.distributed.mooncake_store import Mooncakestore
from vllm_ascend.distributed.mooncake_store_npu import PagedMemNPUConnector, PagedMemNPUConnectorMLA
# First Party


@dataclass
class LasyerBlockReqMeta:
    key: LayerMoonCakeEngineKey
    kvcache: List[torch.Tensor]
    start: int
    end: int
    slot_mapping: torch.Tensor


class MoonCakeEngine:
    #The main class for the cache engine.

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        use_layerwize: bool
    ):
        use_mla = False
        if (
            hasattr(model_config, "use_mla")
            and isinstance(model_config.use_mla, bool)
            and model_config.use_mla
        ):
            use_mla = True
        num_layer = model_config.get_num_layers(parallel_config)
        chunk_size = cache_config.block_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)
        hidden_dim_size = num_kv_head * head_size
        if use_mla:
            kv_shape = (num_layer, 1, chunk_size, 1, head_size)
        else:
            kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
        self.metadata = MoonCakeEngineMetadata(
            model_config.model,
            parallel_config.world_size,
            parallel_config.rank,
            kv_dtype,
            kv_shape,
            chunk_size,
            use_mla,
        )

        self.token_database = ChunkedTokenDatabase(self.metadata)

        self.m_store = Mooncakestore()

        if self.use_mla:
            self.npu_transfer=PagedMemNPUConnectorMLA(hidden_dim_size, num_layer, self.m_store)
        else:
            self.npu_transfer=PagedMemNPUConnector(hidden_dim_size, num_layer, self.m_store)

        if use_layerwize:
            self.load_stream = torch.npu.Stream()
            self.save_stream = torch.npu.Stream()
            self.save_input_queue: queue.Queue[list[LasyerBlockReqMeta]] = queue.Queue()
            self.save_thread = threading.Thread(target=self._save_listener)
            self.save_thread.start()
            self.num_layers = num_layer

    @torch.inference_mode()
    def store(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens and mask into the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)

        for start, end, key in self.token_database.process_tokens(tokens, mask):
            if self.m_store.exists(key):
                continue
            num_tokens = end - start
            kv_shape = self.npu_transfer.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            tensor = torch.empty(kv_shape, dtype=kv_dtype) 
            self.npu_transfer.npu_d2h(tensor, start, end, **kwargs)
            self.m_store.put(key, tensor, kv_shape, kv_dtype)
        logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    @torch.inference_mode()
    def retrieve(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. 

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            memory_tensor = self.m_store.get(key)    
            ret_mask[start:end] = True
            self.npu_transfer.npu_h2d(memory_tensor, start, end, **kwargs)

        retrieved_tokens = torch.sum(ret_mask)
        logger.debug(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )
        return ret_mask
    
    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. 
        """

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
      
        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            keys_multi_layer = key.split_layers(self.num_layers)

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]   # [num_layer,block_num]

            get_generator = self.layerwise_batched_get(keys_layer_major, starts, ends,**kwargs) # load layerwise kv generator
            
            for layer_id in range(self.num_layers):
                next(get_generator)
                yield None

        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        retrieved_tokens = torch.sum(ret_mask)
        logger.debug(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )

        yield ret_mask

    def layerwise_batched_get(
        self,
        keys: List[List[LayerMoonCakeEngineKey]],
        starts: List[int], 
        ends: List[int],
        **kwargs,
    ) -> Generator[None, None, None]:
        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        self.load_stream.synchronize()
        for i,keys_multi_chunk in enumerate(keys):
            for index, key in enumerate(keys_multi_chunk):
                with torch.npu.stream(self.load_stream):
                    memory_tensor = self.m_store.get(key)
                    self.npu_transfer.npu_h2d_layer(memory_tensor, starts[index], ends[index], kvcaches[i], **kwargs)
            yield 

    def store_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)

        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
        
        if keys:
            keys = [list(row) for row in zip(*keys, strict=False)]
            save_generator = self.layerwise_batched_save(keys, starts, ends, **kwargs)
            for layer_id in range(self.num_layers):   
                try:
                    next(save_generator)
                except StopIteration:
                    raise 
                yield
        else:
            for layer_id in range(self.num_layers):
                yield
        logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    def layerwise_batched_save(
        self,
        keys: List[List[LayerMoonCakeEngineKey]],
        starts: List[int], 
        ends: List[int],
        **kwargs,
    ) -> Generator[None, None, None]:
        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        for i,keys_multi_chunk in enumerate(keys):
            req_meta_list:list[LasyerBlockReqMeta]=[]
            for index, key in enumerate(keys_multi_chunk):
                req_meta=LasyerBlockReqMeta(
                    key,
                    kvcaches[i],
                    starts[index],
                    ends[index],
                    slot_mapping
                )
                req_meta_list.append(req_meta)
            self.save_input_queue.put(req_meta_list)
            yield 

    def _save_listener(self):
        kv_dtype = self.metadata.kv_dtype
        while True:
            req_meta_list = self.save_input_queue.get()
            for req_meta in req_meta_list:
                num_tokens=req_meta.end-req_meta.start
                kv_shape = self.npu_transfer.get_layer_shape(num_tokens)
                tensor = torch.empty(kv_shape, dtype=kv_dtype) 
                with torch.npu.stream(self.save_stream):
                    # pass
                    self.npu_transfer.npu_d2h_layer(tensor, req_meta.start, req_meta.end, kvcaches=req_meta.kvcache, slot_mapping=req_meta.slot_mapping)
                    self.m_store.put(req_meta.key, tensor, kv_shape, kv_dtype)
            self.save_stream.synchronize()

    def wait_layer_transfer_finish(self):
        # pass
        while not self.save_input_queue.empty():
            time.sleep(0.0000001)
        self.save_stream.synchronize()

    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]

        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0

        for start, end, key in self.token_database.process_tokens(tokens):
            try:
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    res=self.m_store.exists(keys_multi_layer[-1])
                else:
                    res=self.m_store.exists(key)
                if res:
                    continue
                else:
                    return start
            except Exception as e:
                logger.warning(f"Remote connection failed in contains: {e}")
                return start

        # all tokens where found, return the maximal end
        return end

    def close(self) -> None:
        """Close the cache engine and free all the resources"""      
        self.m_store.close()
