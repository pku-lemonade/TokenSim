from __future__ import annotations

from typing import Any, Iterable
from dataclasses import replace

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError

from .base import BaseKVConnector
from .metadata import (
    ConnectorTransferPlan,
    ConnectorStats,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


class MultiConnector(BaseKVConnector):
    name = "MultiConnector"
    supports_grouped_cache = True

    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
        children: Iterable[BaseKVConnector] | None = None,
    ) -> None:
        super().__init__(config, cache_config)
        self.children = list(children or [])
        self.stats = ConnectorStats()
        self._load_owners = {}

    def set_simulation_time(self, now: float) -> None:
        super().set_simulation_time(now)
        for child in self.children:
            child.set_simulation_time(now)

    def bind_runtime(self, worker):
        for child in self.children:
            child.bind_runtime(worker)

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        loads: list[ConnectorTransferPlan] = []
        saves: list[ConnectorTransferPlan] = []
        preempted: list[int] = []
        for index, child in enumerate(self.children):
            meta = child.build_connector_meta(scheduler_output)
            loads.extend(self._tag(plan, index) for plan in meta.loads)
            saves.extend(self._tag(plan, index) for plan in meta.saves)
            preempted.extend(meta.preempted_request_ids)
        return KVConnectorMetadata(
            connector_name=self.name,
            loads=loads,
            saves=saves,
            preempted_request_ids=preempted,
        )

    def build_transfer_plan(self, **kwargs) -> ConnectorTransferPlan:
        for index, child in enumerate(self.children):
            builder = getattr(child, "build_transfer_plan", None)
            if builder is not None:
                return self._tag(builder(**kwargs), index)
        raise AttributeError(
            "MultiConnector has no child that can build a transfer plan"
        )

    def on_new_request(self, req) -> None:
        for child in self.children:
            child.on_new_request(req)

    def get_num_new_matched_tokens(self, req, num_computed_tokens: int) -> int:
        matches = [
            child.get_num_new_matched_tokens(req, num_computed_tokens)
            for child in self.children
        ]
        tokens = max(matches, default=0)
        owner = matches.index(tokens) if tokens else None
        self._load_owners[req.id] = owner
        for index, child in enumerate(self.children):
            if index != owner:
                child.discard_load(req)
        return tokens

    def update_state_after_alloc(self, req, blocks, num_external_tokens: int) -> None:
        owner = self._load_owners.get(req.id)
        if owner is not None:
            self.children[owner].update_state_after_alloc(
                req, blocks, num_external_tokens
            )

    def discard_load(self, req):
        self._load_owners.pop(req.id, None)
        for child in self.children:
            child.discard_load(req)

    def set_block_releaser(self, releaser):
        for child in self.children:
            setter = getattr(child, "set_block_releaser", None)
            if setter is not None:
                setter(releaser)

    def handle_preemptions(self, meta):
        for child in self.children:
            child.handle_preemptions(meta)
        for req_id in meta.preempted_request_ids:
            self._load_owners.pop(req_id, None)

    def update_connector_output(
        self,
        worker_output: KVConnectorWorkerMetadata,
    ) -> None:
        for child in self.children:
            child.update_connector_output(worker_output)
        super().update_connector_output(worker_output)

    def request_finished(self, req, block_ids):
        self._load_owners.pop(req.id, None)
        delay = False
        payload = {}
        for child in self.children:
            child_delay, child_payload = child.request_finished(req, block_ids)
            delay = delay or child_delay
            if child_payload:
                payload[child.name] = child_payload
        return delay, payload or None

    def on_request_released(self, req) -> None:
        self._load_owners.pop(req.id, None)
        for child in self.children:
            child.on_request_released(req)

    def bind_connector_metadata(self, meta: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(meta)
        routed = [
            KVConnectorMetadata(
                connector_name=child.name,
                preempted_request_ids=list(meta.preempted_request_ids),
            )
            for child in self.children
        ]
        for kind in ("loads", "saves"):
            for plan in getattr(meta, kind):
                path = plan.extra.get("connector_path", ())
                if path:
                    index = path[0]
                else:
                    name = plan.extra.get("connector", meta.connector_name)
                    index = next(
                        (
                            i
                            for i, child in enumerate(self.children)
                            if child.name == name
                        ),
                        None,
                    )
                if index is None or not 0 <= index < len(self.children):
                    raise ConfigurationError(
                        "transfer plan has no matching MultiConnector child"
                    )
                extra = {**plan.extra, "connector_path": tuple(path[1:])}
                getattr(routed[index], kind).append(replace(plan, extra=extra))
        for child, child_meta in zip(self.children, routed):
            child.bind_connector_metadata(child_meta)

    @staticmethod
    def _tag(plan, index):
        return replace(
            plan,
            extra={
                **plan.extra,
                "connector_path": (index, *plan.extra.get("connector_path", ())),
            },
        )

    def start_load_kv(self) -> float:
        return sum(child.start_load_kv() for child in self.children)

    def wait_for_save(self) -> float:
        return sum(child.wait_for_save() for child in self.children)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        meta = super().build_connector_worker_meta()
        for child in self.children:
            meta = meta.aggregate(child.build_connector_worker_meta())
        return meta
