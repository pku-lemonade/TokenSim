"""Uniform block-store transfer plans, cache namespaces and serialized keys."""

from __future__ import annotations

from typing import Any
from functools import cached_property
import json
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import PoolKey, Segment, pool_keys_for_request
from ..metadata import ConnectorTransferPlan


class BlockTransferPlans:
    """Uniform block-store transfer plans, cache namespaces and serialized keys."""

    def _build_load_plan(
        self,
        req: Request,
        load_info: dict[str, Any],
    ) -> ConnectorTransferPlan:
        blocks = int(load_info["blocks"])
        bytes_ = blocks * self.block_bytes
        tier = load_info.get("tier") or "memory"
        transfer = self.transfer_engine.create_transfer(
            source_segment=Segment(
                name=f"store-{tier}", tier="dram" if tier == "memory" else tier
            ),
            target_segment=Segment(name=f"worker-{self.config.kv_rank}", tier="vram"),
            blocks=blocks,
            bytes_=bytes_,
            protocol=self.options.protocol,
            kind="store",
            topology="cross_node",
        )
        pool_keys = list(load_info["keys"][:blocks])
        self._request_keys[req.id] = pool_keys
        keys = [key.to_string() for key in pool_keys]
        return ConnectorTransferPlan(
            request_ids=[req.id],
            request_block_counts={req.id: blocks},
            target_worker_id=self.config.kv_rank,
            blocks=blocks,
            bytes=bytes_,
            latency=self._transfer_latency(transfer),
            kind="load",
            keys=keys,
            tier=tier,
            async_transfer=self.options.load_async,
            extra={
                "protocol": transfer.protocol,
                "connector": self.name,
                "pool_keys": pool_keys,
            },
        )

    def _build_save_plan(
        self, req: Request, keys: list[PoolKey]
    ) -> ConnectorTransferPlan | None:
        computed = min(req.prefill_len, req.num_computed_tokens + req.step_tokens)
        full_blocks = min(len(keys), computed // self.block_size)
        if full_blocks == 0:
            return None
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
        transfer = self.transfer_engine.create_transfer(
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
            latency=self._transfer_latency(transfer),
            kind="save",
            keys=key_strings,
            tier="memory",
            async_transfer=self.options.load_async,
            extra={
                "protocol": transfer.protocol,
                "connector": self.name,
                "pool_keys": pool_keys,
                "request": req,
            },
        )

    def _keys_for_request(self, req: Request) -> list[PoolKey]:
        return self._request_key_bundle(req)[0]

    def _key_strings_for_request(self, req: Request) -> list[str]:
        return self._request_key_bundle(req)[1]

    @cached_property
    def store_namespace(self):
        cache = self.cache_config
        model = getattr(cache, "model_spec", None)
        parallel = getattr(cache, "parallel_config", None)
        shape = [
            self.block_size,
            self.block_bytes,
            getattr(parallel, "tensor_parallel_size", 1),
            getattr(parallel, "pipeline_parallel_size", 1),
            getattr(model, "kv_cache_dtype", None),
            getattr(model, "dtype", None),
            model.cache_signature() if model is not None else None,
        ]
        return "@layout:" + json.dumps(shape, separators=(",", ":"))

    def _request_key_bundle(self, req: Request) -> tuple[list[PoolKey], list[str]]:
        bundle = self._keys_cache.get(req.id)
        if bundle is None:
            keys = pool_keys_for_request(
                req,
                model_name=self.model_name,
                rank_info=self.rank_info,
                store_namespace=self.store_namespace,
                cache_prefix=self.options.cache_prefix,
                tenant_id=self.options.tenant_id,
                group_id=int(self.options.group_id),
                pcp_rank=int(self.options.pcp_rank),
                dcp_rank=int(self.options.dcp_rank),
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
        offload = self.service.store.offload
        return [
            plan
            for plan in plans
            if plan.extra.get("connector") == self.name
            or plan.tier == "memory"
            or (offload is not None and plan.tier == offload.tier)
        ]
