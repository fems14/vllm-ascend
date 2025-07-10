# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Dict, Generator, List, Optional, Union
import asyncio
import multiprocessing
import time

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
from vllm_ascend.distributed.config_data import MoonCakeEngineMetadata, ChunkedTokenDatabase
from vllm_ascend.distributed.mooncake_store import Mooncakestore
from vllm_ascend.distributed.mooncake_store_npu import PagedMemNPUConnector
# First Party


class MoonCakeEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        # token_database: TokenDatabase,
    ):
        # global VLLM_CACHE_CONFIG
        # global VLLM_PARALLEL_CONFIG
        # global VLLM_MODEL_CONFIG
        # global VLLM_SCHEDULER_CONFIG
        # VLLM_CACHE_CONFIG = cache_config
        # VLLM_PARALLEL_CONFIG = parallel_config
        # VLLM_MODEL_CONFIG = model_config
        # VLLM_SCHEDULER_CONFIG = scheduler_config
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
        self.npu_transfer=PagedMemNPUConnector(hidden_dim_size, num_layer)


        self.token_database = ChunkedTokenDatabase(self.metadata)


        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)   #????

        self.m_store = Mooncakestore()

        # InitializeUsageContext(config.to_original_config(), metadata)
        # self.stats_monitor = LMCStatsMonitor.GetOrCreate()

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

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)
        # monitor_req_id = self.stats_monitor.on_store_request(num_stored_tokens)

        for start, end, key in self.token_database.process_tokens(tokens, mask):
            if self.m_store.exists(key):
                continue
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.npu_transfer.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            tensor = torch.empty(kv_shape, dtype=kv_dtype) 
            # self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)  D2H
            self.npu_transfer.npu_d2h(tensor, start, end, **kwargs)
            self.m_store.put(key, tensor, kv_shape, kv_dtype)

        # self.stats_monitor.on_store_finished(monitor_req_id)

        logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    @torch.inference_mode()
    def store_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_name: str =None,
        **kwargs,
    ) -> None:
        """Store the tokens and mask into the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)
        # monitor_req_id = self.stats_monitor.on_store_request(num_stored_tokens)
        for start, end, key in self.token_database.process_tokens(tokens, mask):

            key.chunk_hash=key.chunk_hash+layer_name
            if self.m_store.exists(key):
                continue
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.npu_transfer.get_shape_layer(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            tensor = torch.empty(kv_shape, dtype=kv_dtype)
            # self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)  D2H
            self.npu_transfer.npu_d2h_layer(tensor, start, end, **kwargs)
            self.m_store.put(key, tensor, kv_shape, kv_dtype)

        # self.stats_monitor.on_store_finished(monitor_req_id)

        logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    @torch.inference_mode()
    def retrieve(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        # monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)  monitor is usseful?

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            # Get the memory object from the storage backend
            # try:
            memory_tensor = self.m_store.get(key)    
            # except Exception as e:
            #     logger.warning(f"Error occurred in get: {e}")
         
            ret_mask[start:end] = True
            self.npu_transfer.npu_h2d(memory_tensor, start, end, **kwargs)
            # self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)     need H2D  

        retrieved_tokens = torch.sum(ret_mask)
        # self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.debug(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )
        return ret_mask


    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_name: str =None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        # monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)  monitor is usseful?

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for start, end, key in self.token_database.process_tokens(tokens, mask):
            # Get the memory object from the storage backend
            # try:
            key.chunk_hash=key.chunk_hash+layer_name
            res=self.m_store.exists(key)
            memory_tensor = self.m_store.get(key)
            # except Exception as e:
            #     logger.warning(f"Error occurred in get: {e}")

            ret_mask[start:end] = True
            self.npu_transfer.npu_h2d_layer(memory_tensor, start, end, **kwargs)
            # self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)     need H2D

        retrieved_tokens = torch.sum(ret_mask)
        # self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.debug(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )
        return ret_mask
    # def prefetch(
    #     self,
    #     tokens: torch.Tensor,
    #     mask: Optional[torch.Tensor] = None,
    # ) -> None:
    #     """Launch the prefetching process in the storage manager to load the
    #     KV to the local CPU memory
    #     """
    #     for start, end, key in self.token_database.process_tokens(tokens, mask):
    #         assert isinstance(key, CacheEngineKey)
    #         self.storage_manager.prefetch(key)

    # TODO(Jiayi): Currently, search_range is only used for testing.
    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
        pin: bool = False,
        use_layerwize: bool =False
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: If True, pin the KV cache in the storage.

        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0

        for start, end, key in self.token_database.process_tokens(tokens):
            try:
                if use_layerwize:
                    key.chunk_hash=key.chunk_hash+"model.layers.0.self_attn.attn"
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

    # def clear(
    #     self,
    #     tokens: Optional[Union[torch.Tensor, List[int]]] = None,
    #     locations: Optional[List[str]] = None,
    # ) -> int:
    #     # Clear all caches if tokens is None
    #     if tokens is None or len(tokens) == 0:
    #         num_cleared = self.storage_manager.clear(locations)
    #         return num_cleared

    #     num_removed = 0
    #     # Only remove the caches for the given tokens
    #     for start, end, key in self.token_database.process_tokens(tokens):
    #         # assert isinstance(key, CacheEngineKey)
    #         removed = self.storage_manager.remove(key, locations)
    #         num_removed += removed
    #     return num_removed

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        
        self.m_store.close()

        # if self.lmcache_worker is not None:
        #     self.lmcache_worker.close()

        # self.storage_manager.close()
        # logger.info("LMCacheEngine closed.")

