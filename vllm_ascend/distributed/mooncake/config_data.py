# Standard
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, List, Optional, Tuple, Union
import os
import re

# Third Party
from numpy import array
import torch, torch_npu
import yaml

# First Party

@dataclass
class MoonCakeEngineMetadata:
    """name of the LLM model"""

    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    """ (num_layer, 2, metadata.block_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    block_size: int = 128
    """ whether use MLA"""
    use_mla: bool = False
    
@dataclass(order=True)
class MoonCakeEngineKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> List["LayerMoonCakeEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerMoonCakeEngineKey(
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys   

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return MoonCakeEngineKey(
            parts[0], int(parts[1]), int(parts[2]), parts[3]
        )

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        return {
            "__type__": "CacheEngineKey",
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
        }

    @staticmethod
    def from_dict(d):
        return MoonCakeEngineKey(
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
        )


@dataclass(order=True)
class LayerMoonCakeEngineKey(MoonCakeEngineKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash}@{self.layer_id}"
        )

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        return LayerMoonCakeEngineKey(
            parts[0],
            int(parts[1]),
            int(parts[2]),
            parts[3],
            int(parts[4]),
        )


class ChunkedTokenDatabase():
    def __init__(
        self,
        metadata: Optional[MoonCakeEngineMetadata] = None,
    ):
        self.metadata = metadata

    def _make_key_by_hash(self, chunk_hash: str, layer_id: Optional[int] = None):
        assert self.metadata is not None
        return MoonCakeEngineKey(
            self.metadata.model_name,
            self.metadata.world_size,
            self.metadata.worker_id,
            chunk_hash,
        )

    def _hash(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: str,
    ) -> str:
        # TODO: change it to a more efficient hash function
        if isinstance(tokens, torch.Tensor):
            tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes()
        elif isinstance(tokens, list):
            tokens_bytes = array.array("I", tokens).tobytes()
        return hashlib.sha256(prefix_hash.encode("ascii") + tokens_bytes).hexdigest()

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.metadata.block_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with
                shape [metadata.block_size]
        """
        for i in range(0, len(tokens), self.metadata.block_size):
            yield tokens[i : i + self.metadata.block_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[str]:
        prefix_hash = ''
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[MoonCakeEngineKey, str]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses % self.metadata.block_size != 0:
            raise ValueError(
                "The number of Falses in the mask is not a multiple of the chunk size."
            )
        total_len = len(tokens)

        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)

        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.metadata.block_size
            end_idx = min(start_idx + self.metadata.block_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                if make_key:
                    yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                else:
                    yield start_idx, end_idx, hash_val
