from __future__ import annotations

from typing import Any

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import PoolKey, Segment, pool_keys_for_request
from TokenSim.mooncake.service import get_mooncake_service

from .base import BaseKVConnector
from .metadata import (
    ConnectorTransferPlan,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


class MooncakeStoreConnector(BaseKVConnector):
    name = "MooncakeStoreConnector"

    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__(config, cache_config)
        self.block_bytes = 1
        self.block_size = 1
        self.model_name = "unknown"
        self.rank_info = None
        if cache_config is not None:
            self.block_size = int(cache_config.block_size)
            self.block_bytes = int(
                cache_config.block_size * cache_config.size_per_token
            )
            self.model_name = cache_config.model
            self.rank_info = getattr(cache_config, "rank_info", None)
        self.service = get_mooncake_service(config, block_bytes=self.block_bytes)
        self.mooncake_stats = self.service.store.stats
        # "mooncake" mirrors the reference vLLM MooncakeStoreConnector
        # (ref/vllm .../kv_connector/v1/mooncake/store/); "every_step" keeps
        # the historical TokenSim behavior.
        self._aligned = self.service.config.save_policy == "mooncake"
        self._pending_loads: dict[int, dict[str, Any]] = {}
        self._request_keys: dict[int, list[PoolKey]] = {}
        # Per-request key chain and its string form, both computed exactly once.
        self._keys_cache: dict[int, tuple[list[PoolKey], list[str]]] = {}
        self._saved_request_ids: set[int] = set()
        self._pending_async_puts: set[int] = set()
        self._delayed_releases: dict[int, tuple[Request, list[int]]] = {}
        self._block_releaser: Any | None = None

    def get_num_new_matched_tokens(
        self,
        req: Request,
        num_computed_tokens: int,
    ) -> int:
        keys = self._keys_for_request(req)
        self._request_keys[req.id] = keys
        if not keys:
            return 0
        local_gpu_tokens = max(0, int(num_computed_tokens))
        if local_gpu_tokens:
            self.mooncake_stats.local_gpu_hit_tokens += local_gpu_tokens
        hit = self.service.store.lookup(keys, now=self.simulation_time)
        if hit.hit_blocks <= 0:
            return 0
        hit_keys = list(hit.keys)
        hit_tokens = hit.hit_blocks * self.block_size
        if self._aligned and hit_tokens >= req.prefill_len:
            # Reference behavior: on a full-prompt hit, leave the trailing
            # block uncomputed-from-cache so prefill still computes tokens.
            capped_blocks = max(0, (req.prefill_len - 1) // self.block_size)
            hit_keys = hit_keys[:capped_blocks]
            hit_tokens = capped_blocks * self.block_size
        external_tokens = max(0, hit_tokens - local_gpu_tokens)
        if external_tokens <= 0:
            return 0
        if hit.tier == "ssd":
            self.mooncake_stats.mooncake_disk_hit_tokens += external_tokens
        else:
            self.mooncake_stats.mooncake_memory_hit_tokens += external_tokens
        self._pending_loads[req.id] = {
            "keys": hit_keys,
            "tier": hit.tier,
            "tokens": external_tokens,
            "blocks": external_tokens // self.block_size,
        }
        return external_tokens

    def set_block_releaser(self, releaser) -> None:
        self._block_releaser = releaser

    def update_state_after_alloc(
        self,
        req: Request,
        blocks: list[Any],
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens:
            req.cached_prefill_tokens = max(
                req.cached_prefill_tokens,
                num_external_tokens,
            )
            req.cached_prefill_blocks = max(
                req.cached_prefill_blocks,
                num_external_tokens // self.block_size,
            )
            req.effective_prefill_tokens = max(
                0,
                req.prefill_len - req.cached_prefill_tokens,
            )
            req.reuse_hit_blocks = max(req.reuse_hit_blocks, req.cached_prefill_blocks)
            req.reuse_miss_blocks = max(
                0,
                req.prefill_len // self.block_size - req.reuse_hit_blocks,
            )

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        loads: list[ConnectorTransferPlan] = []
        saves: list[ConnectorTransferPlan] = []
        can_save = self.config.kv_role in {"kv_producer", "kv_both", None}
        for req in getattr(scheduler_output, "scheduled", []) or []:
            load_info = self._pending_loads.pop(req.id, None)
            if load_info is not None:
                loads.append(self._build_load_plan(req, load_info))
            if not can_save:
                continue
            if self._aligned:
                # Reference semantics: saves happen only within the prefill
                # range (decode steps never save), each request saves at most
                # once, and a step that loads from the store skips its save.
                if not req.is_prefill or req.id in self._saved_request_ids:
                    continue
                self._saved_request_ids.add(req.id)
                if load_info is not None:
                    continue
            save_keys = self._keys_for_request(req)
            if not save_keys:
                continue
            plan = self._build_save_plan(req, save_keys)
            if plan is None:
                continue
            saves.append(plan)
            if plan.async_transfer:
                self._pending_async_puts.update(plan.request_ids)
        preempted = [req.id for req in getattr(scheduler_output, "preempted", []) or []]
        return KVConnectorMetadata(
            connector_name=self.name,
            loads=loads,
            saves=saves,
            preempted_request_ids=preempted,
        )

    def start_load_kv(self) -> float:
        latency = 0.0
        for plan in self._own_plans(self._metadata.loads):
            keys = self._pool_keys_from_plan(plan)
            disk_latency = self.service.store.get(
                keys,
                plan.tier,
                now=self.simulation_time,
            )
            latency += plan.latency + disk_latency
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=plan.latency + disk_latency,
                kind="load",
            )
            self.mooncake_stats.record_transfer(
                bytes_=plan.bytes,
                latency=plan.latency + disk_latency,
                kind="store_load",
                blocking_latency=(
                    0.0
                    if plan.async_transfer and self.service.config.transfer_overlap
                    else plan.latency + disk_latency
                ),
            )
            self.mooncake_stats.record_pool_keys(plan.keys)
        if self.service.config.load_async and self.service.config.transfer_overlap:
            self.mooncake_stats.pending_async_jobs += len(self._metadata.loads)
            return 0.0
        return latency

    def wait_for_save(self) -> float:
        latency = 0.0
        for plan in self._own_plans(self._metadata.saves):
            keys = self._pool_keys_from_plan(plan)
            store_timing = self.service.store.put_with_timing(
                keys,
                blocks=plan.blocks,
                now=self.simulation_time,
                initial_delay=plan.latency,
            )
            total_latency = plan.latency + store_timing.accounting_latency
            blocking_latency = plan.latency + store_timing.blocking_latency
            latency += blocking_latency
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=total_latency,
                kind="save",
                blocking_latency=blocking_latency,
            )
            self.mooncake_stats.record_transfer(
                bytes_=plan.bytes,
                latency=total_latency,
                kind="store_save",
                blocking_latency=(
                    0.0
                    if plan.async_transfer and self.service.config.transfer_overlap
                    else blocking_latency
                ),
            )
            self.mooncake_stats.record_pool_keys(plan.keys)
        if self.service.config.load_async and self.service.config.transfer_overlap:
            self.mooncake_stats.pending_async_jobs += len(self._metadata.saves)
            return 0.0
        return latency

    def request_finished(
        self,
        req: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if req.id in self._pending_async_puts:
            self._delayed_releases[req.id] = (req, list(block_ids))
            return True, {"pending_async_put": True}
        self._forget_request(req.id)
        return False, None

    def get_finished(
        self,
        finished_req_ids: set[int],
    ) -> tuple[set[int], set[int]]:
        finished_sending = set(finished_req_ids) & self._pending_async_puts
        finished_recving = set(finished_req_ids) & self._pending_async_puts
        return finished_sending, finished_recving

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        meta = super().build_connector_worker_meta()
        if self._pending_async_puts:
            meta.finished_sending.update(self._pending_async_puts)
            meta.finished_recving.update(self._pending_async_puts)
        return meta

    def update_connector_output(
        self,
        worker_output: KVConnectorWorkerMetadata,
    ) -> None:
        releasable = set(worker_output.finished_sending)
        self._pending_async_puts.difference_update(worker_output.finished_sending)
        self.mooncake_stats.pending_async_jobs = max(
            0,
            self.mooncake_stats.pending_async_jobs
            - len(worker_output.finished_sending),
        )
        for req_id in releasable:
            delayed = self._delayed_releases.pop(req_id, None)
            if delayed is None or self._block_releaser is None:
                self._forget_request(req_id)
                continue
            req, block_ids = delayed
            self._block_releaser(req, block_ids)
            self._forget_request(req_id)
        super().update_connector_output(worker_output)

    def handle_preemptions(self, meta: KVConnectorMetadata) -> None:
        super().handle_preemptions(meta)
        # Recomputed requests rebuild connector state when they are admitted again.
        for req_id in meta.preempted_request_ids:
            self._forget_request(req_id)

    def on_request_released(self, req: Request) -> None:
        if req.id in self._pending_async_puts:
            # An async save is still in flight; the delayed-release path in
            # update_connector_output() cleans up once it completes.
            return
        self._forget_request(req.id)

    def _forget_request(self, req_id: int) -> None:
        self._keys_cache.pop(req_id, None)
        self._request_keys.pop(req_id, None)
        self._pending_loads.pop(req_id, None)
        self._saved_request_ids.discard(req_id)

    def _build_load_plan(
        self,
        req: Request,
        load_info: dict[str, Any],
    ) -> ConnectorTransferPlan:
        blocks = int(load_info["blocks"])
        bytes_ = blocks * self.block_bytes
        tier = load_info.get("tier") or "memory"
        transfer = self.service.transfer_engine.create_transfer(
            source_segment=Segment(
                name=f"store-{tier}", tier="ssd" if tier == "ssd" else "dram"
            ),
            target_segment=Segment(name=f"worker-{self.config.kv_rank}", tier="vram"),
            blocks=blocks,
            bytes_=bytes_,
            protocol="nvmeof" if tier == "ssd" else self.service.config.protocol,
            kind="store",
            topology="cross_node",
        )
        pool_keys = list(load_info["keys"][:blocks])
        self._request_keys[req.id] = pool_keys
        keys = self._key_strings_for_request(req)[: len(pool_keys)]
        return ConnectorTransferPlan(
            request_ids=[req.id],
            request_block_counts={req.id: blocks},
            target_worker_id=self.config.kv_rank,
            blocks=blocks,
            bytes=bytes_,
            latency=transfer.latency,
            kind="load",
            keys=keys,
            tier=tier,
            async_transfer=self.service.config.load_async,
            extra={
                "protocol": transfer.protocol,
                "connector": self.name,
                "pool_keys": pool_keys,
            },
        )

    def _build_save_plan(
        self, req: Request, keys: list[PoolKey]
    ) -> ConnectorTransferPlan | None:
        full_blocks = min(len(keys), req.prefill_len // self.block_size)
        pool_keys = keys[:full_blocks]
        key_strings = self._key_strings_for_request(req)[:full_blocks]
        if self._aligned:
            # Reference semantics: an existence prefilter runs before the put
            # (worker-side batch_is_exist) so only missing keys are written
            # and transferred.
            missing = self.service.store.missing_indices(pool_keys)
            if not missing:
                self._request_keys[req.id] = pool_keys
                return None
            pool_keys = [pool_keys[i] for i in missing]
            key_strings = [key_strings[i] for i in missing]
        self._request_keys[req.id] = pool_keys
        blocks = len(pool_keys)
        bytes_ = blocks * self.block_bytes
        transfer = self.service.transfer_engine.create_transfer(
            source_segment=Segment(name=f"worker-{self.config.kv_rank}", tier="vram"),
            target_segment=Segment(name="store-memory", tier="dram"),
            blocks=blocks,
            bytes_=bytes_,
            kind="store",
            topology="cross_node",
        )
        return ConnectorTransferPlan(
            request_ids=[req.id],
            request_block_counts={req.id: blocks},
            source_worker_id=self.config.kv_rank,
            blocks=blocks,
            bytes=bytes_,
            latency=transfer.latency,
            kind="save",
            keys=key_strings,
            tier="memory",
            async_transfer=self.service.config.load_async,
            extra={
                "protocol": transfer.protocol,
                "connector": self.name,
                "pool_keys": pool_keys,
            },
        )

    def _keys_for_request(self, req: Request) -> list[PoolKey]:
        return self._request_key_bundle(req)[0]

    def _key_strings_for_request(self, req: Request) -> list[str]:
        return self._request_key_bundle(req)[1]

    def _request_key_bundle(self, req: Request) -> tuple[list[PoolKey], list[str]]:
        bundle = self._keys_cache.get(req.id)
        if bundle is None:
            keys = pool_keys_for_request(
                req,
                model_name=self.model_name,
                rank_info=self.rank_info,
                engine_id=self.config.engine_id or "default",
                group_id=int(self.service.config.group_id),
                pcp_rank=int(self.service.config.pcp_rank),
                dcp_rank=int(self.service.config.dcp_rank),
            )
            bundle = (keys, [key.to_string() for key in keys])
            self._keys_cache[req.id] = bundle
        return bundle

    def _pool_keys_from_plan(self, plan: ConnectorTransferPlan) -> list[PoolKey]:
        pool_keys = plan.extra.get("pool_keys")
        if pool_keys is not None:
            return list(pool_keys)
        # Fallback for plans built elsewhere; only live requests are indexed.
        key_lookup = {
            key.to_string(): key for keys in self._request_keys.values() for key in keys
        }
        result = []
        for key_string in plan.keys:
            key = key_lookup.get(key_string)
            if key is not None:
                result.append(key)
        return result

    def _own_plans(
        self, plans: list[ConnectorTransferPlan]
    ) -> list[ConnectorTransferPlan]:
        return [
            plan
            for plan in plans
            if plan.extra.get("connector") == self.name
            or plan.tier in {"memory", "ssd"}
        ]
