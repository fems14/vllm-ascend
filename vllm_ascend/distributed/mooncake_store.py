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
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, no_type_check
import asyncio
import json
import operator
import os
import struct
import ctypes

# Third Party
import torch, torch_npu

# First Party
from vllm.utils import logger
from vllm.config import VllmConfig
from vllm_ascend.distributed.config_data import MoonCakeEngineKey

METADATA_BYTES_LEN = 24


DTYPE_TO_INT = {
    None: 0,
    torch.half: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float: 4,
    torch.float32: 4,
    torch.float64: 5,
    torch.double: 5,
    torch.uint8: 6,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 8,
}

INT_TO_DTYPE = {
    0: None,
    1: torch.half,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.float,
    5: torch.float64,
    6: torch.uint8,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}

@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)


class Mooncakestore():
    def __init__(
        self,
    ):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
            print(f"protocol:{ self.config.protocol}")
            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol, self.config.device_name,
                             self.config.master_server_address)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def exists(self, key: MoonCakeEngineKey) -> bool:
        print(f"dfq get key:{key.to_string()}, exist:{self.store.is_exist(key.to_string())}")
        return self.store.is_exist(key.to_string())

    def get(self, key: MoonCakeEngineKey) -> Optional[torch.Tensor]:  #to be amend
        key_str = key.to_string()
        print(f"dfq get key:{key_str}")
        try:
            buffer = self.store.get_buffer(key_str)
        except Exception as e:
            logger.error(f"Failed to get key {key_str}. {e}")

        if buffer is None:
            return None

        retrieved_view = memoryview(buffer)   
        metadata_bytes = retrieved_view[:METADATA_BYTES_LEN]
        if metadata_bytes is None or len(metadata_bytes) != METADATA_BYTES_LEN:
            return None

        length, dtype, shape0, shape1, shape2, shape3 = struct.unpack_from(
            "iiiiii", metadata_bytes
        ) 
         
        shape=torch.Size([shape0, shape1, shape2, shape3])

        num_elements = reduce(operator.mul, shape)
        temp_tensor = torch.frombuffer(
                buffer,
                dtype=INT_TO_DTYPE[dtype],
                offset=METADATA_BYTES_LEN,
                count=num_elements,
            ).reshape(shape)
        return temp_tensor

    def put(self, key: MoonCakeEngineKey, memory_tebsor: torch.Tensor, shape:torch.Size, dtype:torch.dtype):   #to be amend
        # Please use a function like `memory_obj.to_meta()`.
        num_bytes = memory_tebsor.numel() * memory_tebsor.element_size()
        ptr = memory_tebsor.data_ptr()
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents)
        )
        kv_bytes=memoryview(byte_array)

        metadata_bytes=struct.pack(
            "iiiiii",
            len(kv_bytes),
            DTYPE_TO_INT[dtype],
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        assert len(metadata_bytes) == METADATA_BYTES_LEN
        key_str = key.to_string()
        print(f"dfq put key:{key_str}, len:{len(kv_bytes)}")
        try:
            self.store.put_parts(key_str, metadata_bytes, kv_bytes)
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str},"
                f"meta type: {type(metadata_bytes)},"
                f"data: {type(kv_bytes)}: {e}"
            )
        print(f"dfq put key:{key_str}, len:{len(kv_bytes)}")

    @no_type_check
    def list(self) -> List[str]:
        pass

    def close(self):
        self.store.close()
        logger.info("Closed the mooncake store connection")
