
import threading
from enum import Enum
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Dict, List, Tuple, Union
import msgspec
import torch
import zmq

from concurrent.futures import Future

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.utils import logger
from vllm.utils import make_zmq_path, make_zmq_socket, round_down, get_ip,cdiv
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_ascend.distributed.mooncake.mooncake_engine import MoonCakeEngine
from vllm.v1.request import Request
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in mooncake
    mooncake_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool

@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=0,
        )

    # def update(
    #     self,
    #     cached_request: "CachedRequestData",
    # ) -> None:
    #     """Update the request tracker when a running request is
    #     scheduled again
    #     """
    #     self.token_ids.extend(cached_request.new_token_ids)
    #     new_block_ids: list[int]
    #     if not isinstance(cached_request.new_block_ids[0], list):
    #         new_block_ids = cached_request.new_block_ids
    #     else:
    #         new_block_ids = cached_request.new_block_ids[0]
    #     self.allocated_block_ids.extend(new_block_ids)
    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[tuple[list[int], ...], list[int]],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: Optional[LoadSpec] = None,
        skip_save: bool = False,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, block_size) * block_size
        )
        skip_save = skip_save or (
            tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (
            (input_token_len // block_size * block_size)
            if discard_partial_chunks
            else input_token_len
        )

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        # OPTIMIZATION: pre-allocate the buffer for token ids and block ids
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]
        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.mooncake_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_spec=save_spec,
            load_spec=load_spec,
        )


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class MooncakeConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise=vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeConnectorV1Scheduler(vllm_config, self.skip_last_n_tokens) 
        else:
            self.connector_worker = MoonCakeEngine(
                vllm_config.model_config,
                vllm_config.parallel_config,
                vllm_config.cache_config,
                vllm_config.scheduler_config,
                self.use_layerwise,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = MooncakeLookupServer(
                    self.connector_worker, vllm_config, self.use_layerwise
                )
    
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]
    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    ############################################################
    # Worker Side Methods
    ############################################################
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        self.current_layer = 0
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return
        
        metadata = self._get_connector_metadata()
        self.layerwise_retrievers = []
        for request in metadata.requests:
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.npu()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._block_size
                * self._block_size
            )
            token_mask[:masked_token_count] = False

            if self.skip_last_n_tokens > 0:
                tokens = tokens[: -self.skip_last_n_tokens]
                token_mask = token_mask[: -self.skip_last_n_tokens]
            if self.use_layerwise:
                layerwise_retriever = self.connector_worker.retrieve_layer(
                    tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                )
                next(layerwise_retriever)   # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.connector_worker.retrieve(
                    tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    request.load_spec.mooncake_cached_tokens
                    - request.load_spec.vllm_cached_tokens
                    - self.skip_last_n_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!"
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        if not self.use_layerwise:
            return
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MooncakeConnector does not save explicitly."""
        if not self.use_layerwise:
            return
        
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        
        connector_metadata = self._get_connector_metadata()
       
        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())
        if self.current_layer == 0:
            self.layerwise_storers = []
            for request in connector_metadata.requests:
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                assert isinstance(token_ids, torch.Tensor)
                assert token_ids.is_cpu

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.npu()
                skip_leading_tokens = max(
                    self.connector_worker.lookup(token_ids, self.use_layerwise),
                    save_spec.skip_leading_tokens,
                )
                if skip_leading_tokens == len(token_ids):
                    continue  # skip this request

                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._block_size
                    * self._block_size
                )

                store_mask = torch.ones_like(token_ids, dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False
                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                layerwise_storer = self.connector_worker.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                )
                self.layerwise_storers.append(layerwise_storer)
        for layerwise_storer in self.layerwise_storers:
            try:
                next(layerwise_storer)
            except Exception as e:
                raise
            self.current_layer = self.current_layer + 1

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        
        self.save_layer = 0
        if self.use_layerwise:
            self.connector_worker.wait_layer_transfer_finish()
            return
        
        connector_metadata = self._get_connector_metadata()
       
        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())
        self.layerwise_storers = []

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu  # why is cpu?

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.npu()
            skip_leading_tokens = max(
                self.connector_worker.lookup(token_ids, self.use_layerwise),
                save_spec.skip_leading_tokens,
            )
            if skip_leading_tokens == len(token_ids):
                continue  # skip this request

            skip_leading_tokens = (
                skip_leading_tokens
                // self._block_size
                * self._block_size
            )

            store_mask = torch.ones_like(token_ids, dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False
            
            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            self.connector_worker.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
            )

def get_zmq_rpc_path_mooncake(
    vllm_config: Optional["VllmConfig"] = None,
) -> str:
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "mooncake_rpc_port", 0
        )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/mooncake_rpc_port_{rpc_port}"



class MooncakeConnectorV1Scheduler:
    def __init__(self, vllm_config: "VllmConfig", skip_last_n_tokens):
        self.client=MooncakeLookupClient(vllm_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
                # request_id -> (vllm cached tokes, mooncake cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.skip_last_n_tokens = skip_last_n_tokens
        self._block_size = vllm_config.cache_config.block_size
                # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
                # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
        )
        self._unfinished_requests: dict[str, Request] = {}
    
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """

        token_ids = torch.tensor(request.prompt_token_ids)
        if self.skip_last_n_tokens > 0:
            num_external_hit_tokens = self.client.lookup(
                token_ids[: -self.skip_last_n_tokens]
            )
        else:
            num_external_hit_tokens = self.client.lookup(token_ids)

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, mooncake hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )
        
        if need_to_allocate <= 0:
            return 0, False
        
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            mooncake_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )
        return need_to_allocate, False

    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].mooncake_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].mooncake_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        self._unfinished_requests[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer"

        meta = MooncakeConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            request_tracker = RequestTracker.from_new_request(
                request, num_tokens_to_compute
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # for request in scheduler_output.scheduled_cached_reqs:
        #     request_tracker = self._request_trackers[request.req_id]
        #     request_tracker.update(request)

        #     req_meta = ReqMeta.from_request_tracker(
        #         request_tracker,
        #         self._block_size,
        #         load_spec=None,
        #         skip_save=force_skip_save,
        #         discard_partial_chunks=self._discard_partial_chunks,
        #     )
        #     if req_meta is not None:
        #         meta.add_request(req_meta)
        
        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=None,
                    skip_save=force_skip_save,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        else:
            for i, req_id in enumerate(cached_reqs.req_ids):
                request_tracker = self._request_trackers[req_id]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                if request := self._unfinished_requests.get(req_id):
                    num_current_tokens = len(request_tracker.token_ids)
                    new_token_ids = request.all_token_ids[
                        num_current_tokens : num_current_tokens + num_new_tokens
                    ]
                else:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests, "
                        f"but it is scheduled to be cached"
                    )
                new_block_ids = cached_reqs.new_block_ids[i]
                request_tracker.update(new_token_ids, new_block_ids)
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=None,
                    skip_save=force_skip_save,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        return meta


class MooncakeLookupClient:
    def __init__(self, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_mooncake(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, token_ids: torch.Tensor) -> int:
        request = self.encoder.encode(token_ids)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


class MooncakeLookupServer:
    def __init__(
        self,
        mooncake_engine: MoonCakeEngine,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_mooncake(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.mooncake_engine = mooncake_engine
        self.running = True

        def process_request():
            while self.running:
                frames = self.socket.recv_multipart(copy=False)
                token_ids = self.decoder.decode(frames)
                result = self.mooncake_engine.lookup(token_ids, use_layerwise)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
