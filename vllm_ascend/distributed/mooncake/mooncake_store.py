# Standard
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, no_type_check
from enum import Enum
import asyncio
import json
import operator
import os
import struct
import ctypes
import time
# Third Party
import torch, torch_npu
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.distributed.parallel_state import (get_dp_group,
                                             get_tensor_model_parallel_rank,
                                             get_tp_group)

# First Party
from vllm.utils import logger
from vllm.config import VllmConfig
from vllm_ascend.distributed.mooncake.config_data import MoonCakeEngineKey

METADATA_BYTES_LEN = 24
BASE_PORT = int(os.getenv("VLLM_BASE_PORT", "8790"))


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3


class Mooncakestore():
    def __init__(
        self, parallel_config: ParallelConfig
    ):
        try:
            from pymmc import DistributedObjectStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e
        try:
            self.rank = int(parallel_config.rank)
            self.store = DistributedObjectStore()
            res = self.store.init(int(parallel_config.rank))
            print(f"res:{res}")
            assert res==0
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def exists(self, key: MoonCakeEngineKey) -> bool:
        return self.store.is_exist(key.to_string())
    def set_kv_caches(self, kvcache):
        self.kvcache = list(kvcache)
        print(f"dfq kvcache:{type(self.kvcache)}, {type(self.kvcache[0])}")

    def get(self, key: MoonCakeEngineKey, addr: list[int], size: list[int],block_id:int):
        key_str = key.to_string()
        try:
            res=self.store.get_into_layers(key.to_string(), addr, size, MmcDirect.COPY_G2L.value)
            print(f"key:{key.to_string()} block_id:{block_id}")
            print(f"get kv:{self.kvcache[0][0][block_id]}")
            print(f"res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {key_str}. {e}")

    def put(self, key: MoonCakeEngineKey, addr: List[int], size: List[int], block_id:int):
        try:
            # if self.rank==0:
            #     print(f"dfq sleep")
            #     time.sleep(5)
            res=self.store.put_from_layers(key.to_string(), addr, size, MmcDirect.COPY_L2G.value)
            print(f"key:{key.to_string()} block_id:{block_id}")
            print(f"put kv:{self.kvcache[0][0][block_id]}")
            print(f"res:{res}")
        except Exception as e:
            logger.error(
                f"Failed to put key {key.to_string()},e:{e}"
            )
    

    def close(self):
        self.store.close()
        logger.info("Closed the mooncake store connection")
