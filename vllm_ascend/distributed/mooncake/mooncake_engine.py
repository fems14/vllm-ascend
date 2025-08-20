# Standard
from typing import Dict, Generator, List, Optional, Union
import math
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
from vllm_ascend.distributed.mooncake.config_data import MoonCakeEngineKey, MoonCakeEngineMetadata, ChunkedTokenDatabase, LayerMoonCakeEngineKey
from vllm_ascend.distributed.mooncake.mooncake_store import Mooncakestore
# First Party


@dataclass
class LasyerMultiBlockReqMeta:
    keys: List[LayerMoonCakeEngineKey]
    starts: List[int]
    ends: list[int]
    slot_mapping: torch.Tensor
    layer_id: int


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
        self.use_mla = False
        if (
            hasattr(model_config, "use_mla")
            and isinstance(model_config.use_mla, bool)
            and model_config.use_mla
        ):
            self.use_mla = True
        # self.use_mla = first_kv_cache_tuple[0].size(
        #     -1) != first_kv_cache_tuple[1].size(-1)
        num_layer = model_config.get_num_layers(parallel_config)
        self.block_size = cache_config.block_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)
        self.hidden_dim_size = num_kv_head * head_size
        if self.use_mla:
            kv_shape = (num_layer, 1, self.block_size, 1, head_size)
        else:
            kv_shape = (num_layer, 2, self.block_size, num_kv_head, head_size)
        self.metadata = MoonCakeEngineMetadata(
            model_config.model,
            parallel_config.world_size,
            parallel_config.rank,
            kv_dtype,
            kv_shape,
            self.block_size,
            self.use_mla,
        )

        self.token_database = ChunkedTokenDatabase(self.metadata)

        self.m_store = Mooncakestore(parallel_config)

        if use_layerwize:
            self.load_stream = torch.npu.Stream()
            self.save_stream = torch.npu.Stream()
            self.save_input_queue: queue.Queue[list[LasyerMultiBlockReqMeta]] = queue.Queue()
            self.save_output_queue: queue.Queue[list[LasyerMultiBlockReqMeta]] = queue.Queue()
            self.save_thread = threading.Thread(target=self._save_listener)
            self.load_thread = threading.Thread(target=self._load_listener)
            self.save_thread.start()
            self.load_thread.start()
            self.get_event = threading.Event()
            self.num_layers = num_layer
      
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # TODO(tms): Find a more robust way to detect and handle MLA
        if self.use_mla:
            # MLA case.[num_block, block_size, 1, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 3  # [block_size, latent_dim]
            block_shape_norm = first_kv_cache_tuple[0].shape[-block_rank:]
            block_shape_pe = first_kv_cache_tuple[1].shape[-block_rank:]
            self.block_len = [
                first_kv_cache[0].element_size() * math.prod(block_shape_norm),
                first_kv_cache[1].element_size() * math.prod(block_shape_pe)
            ]
            logger.info(
                "num_blocks: %s, block_shape_norm: %s, block_shape_pe: %s",
                self.num_blocks, block_shape_norm, block_shape_pe)
        else:
            # [num_block, block_size, num_head, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            kv_elem_size = first_kv_cache.element_size()
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            print(f"dfq block shape:{block_shape}")
            self.block_len = [kv_elem_size * math.prod(block_shape)]
            logger.info("num_blocks: %s, block_shape: %s", self.num_blocks,
                        block_shape)

        logger.info("Registering KV_Caches. use_mla: %s, shape %s",
                    self.use_mla, first_kv_cache.shape)

        self.kv_caches = kv_caches
        self.m_store.set_kv_caches(kv_caches.values())
        self.kv_caches_base_addr = []
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            if self.use_mla:
                for i, cache in enumerate(cache_or_caches, 0):
                    base_addr = cache.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)
            else:
                cache_list = [cache_or_caches
                              ] if self.use_mla else cache_or_caches
                for cache in cache_list:
                    base_addr = cache.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)

    def prepare_value(self, start: int, end: int, **kwargs):
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        addr_list=[]
        size_list=[]
        block_id=slot_mapping[start] / self.block_size
        for index, base_addr in enumerate(self.kv_caches_base_addr):   
            block_len = (self.block_len[index % 2]
                        if self.use_mla else self.block_len[0])
 
            addr=base_addr+int(block_id.item())*block_len
            length=int(block_len/self.block_size*(end-start))
            addr_list.append(addr)
            size_list.append(length)
        print(f"start:{start}, end:{end}")
        return addr_list, size_list, int(block_id.item())
    
    def prepare_value_layer(self, start: int, end: int, slot_mapping: torch.Tensor, layer_id: int):
        block_id=slot_mapping[start] / self.block_size
        if self.use_mla:
            addr_k=self.kv_caches_base_addr[layer_id*2]+int(block_id.item())*self.block_len[0]
            addr_v=self.kv_caches_base_addr[layer_id*2+1]+int(block_id.item())*self.block_len[1]
            length_k=int(self.block_len[0]/self.block_size*(end-start))
            length_v=int(self.block_len[1]/self.block_size*(end-start))
            size_list=[length_k, length_v]
        else:
            addr_k=self.kv_caches_base_addr[layer_id*2]+int(block_id.item())*self.block_len[0]
            addr_v=self.kv_caches_base_addr[layer_id*2+1]+int(block_id.item())*self.block_len[0]
            length=int(self.block_len[0]/self.block_size*(end-start))
            size_list=[length, length]
        addr_list=[addr_k,addr_v]
        return addr_list, size_list
    
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
            addr, size, block_id=self.prepare_value(start, end, **kwargs)
            self.m_store.put(key, addr, size, block_id)
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
            addr, size, block_id=self.prepare_value(start, end, **kwargs)
            self.m_store.get(key, addr, size, block_id)    
            ret_mask[start:end] = True

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
        first_flag= True
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys = [list(row) for row in zip(*keys, strict=False)]   # [num_layer,block_num]
            slot_mapping: torch.Tensor = kwargs["slot_mapping"]
            for layer_id, keys_multi_chunk in enumerate(keys):
                if not first_flag:
                    is_finish=self.get_event.wait(timeout=3)
                    if not is_finish:
                        raise SystemError("Layerwise get failed")
                    self.get_event.clear()
                req_meta=LasyerMultiBlockReqMeta(
                        keys_multi_chunk,
                        starts,
                        ends,
                        slot_mapping,
                        layer_id
                    )
                self.save_output_queue.put(req_meta)
                first_flag=False
                yield None
            # get_generator = self.layerwise_batched_get(keys_layer_major, starts, ends,**kwargs) # load layerwise kv generator
            
            # for layer_id in range(self.num_layers):
            #     next(get_generator)
            #     yield None
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

    # def layerwise_batched_get(
    #     self,
    #     keys: List[List[LayerMoonCakeEngineKey]],
    #     starts: List[int], 
    #     ends: List[int],
    #     **kwargs,
    # ) -> Generator[None, None, None]:
    #     kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
    #     self.load_stream.synchronize()
    #     for i,keys_multi_chunk in enumerate(keys):

    #         for index, key in enumerate(keys_multi_chunk):
    #             addr, size=self.prepare_value_layer(starts[index], ends[index],**kwargs)
    #             with torch.npu.stream(self.load_stream):
    #                 memory_tensor = self.m_store.get(key, self.use_mla)
    #         self.npu_transfer.npu_h2d_layer(memory_tensor, starts[index], ends[index], kvcaches[i], **kwargs)
    #         yield 

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
            keys = [list(row) for row in zip(*keys, strict=False)]   # [num_layer,block_num]
            slot_mapping: torch.Tensor = kwargs["slot_mapping"]
            for layer_id, keys_multi_chunk in enumerate(keys):
                req_meta=LasyerMultiBlockReqMeta(
                        keys_multi_chunk,
                        starts,
                        ends,
                        slot_mapping,
                        layer_id
                    )
                self.save_input_queue.put(req_meta)
                yield
            # save_generator = self.layerwise_batched_save(keys, starts, ends, **kwargs)
            # for layer_id in range(self.num_layers):   
            #     try:
            #         next(save_generator)
            #     except StopIteration:
            #         raise 
            #     yield
        else:
            for layer_id in range(self.num_layers):
                yield
        logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    # def layerwise_batched_save(
    #     self,
    #     keys: List[List[LayerMoonCakeEngineKey]],
    #     starts: List[int], 
    #     ends: List[int],
    #     **kwargs,
    # ) -> Generator[None, None, None]:
    #     slot_mapping: torch.Tensor = kwargs["slot_mapping"]
    #     for keys_multi_chunk in enumerate(keys):
    #         req_meta=LasyerMultiBlockReqMeta(
    #                 keys_multi_chunk,
    #                 starts,
    #                 ends,
    #                 slot_mapping
    #             )
    #         self.save_input_queue.put(req_meta)
    #         yield 

    def _save_listener(self):
        while True:
            req_meta_chunck = self.save_input_queue.get()
            torch.npu.current_stream().synchronize()
            for index, key in enumerate(req_meta_chunck.keys):
                addr, size=self.prepare_value_layer(req_meta_chunck.starts[index], req_meta_chunck.ends[index], req_meta_chunck.slot_mapping, req_meta_chunck.layer_id)
                self.m_store.put(key, addr, size)
    
    def _load_listener(self):
        while True:
            req_meta_chunck = self.save_output_queue.get()
            for index, key in enumerate(req_meta_chunck.keys):
                addr, size=self.prepare_value_layer(req_meta_chunck.starts[index], req_meta_chunck.ends[index], req_meta_chunck.slot_mapping, req_meta_chunck.layer_id)
                self.m_store.get(key, addr, size)
            self.get_event.set()
            

    def wait_layer_transfer_finish(self):
        time.sleep(1)
        # pass
        # while not self.save_input_queue.empty():
        #     time.sleep(0.0000001)
        # self.save_stream.synchronize()

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
