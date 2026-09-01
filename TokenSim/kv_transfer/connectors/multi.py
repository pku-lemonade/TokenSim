from __future__ import annotations

from typing import Any, Iterable

from TokenSim.config.config import CacheConfig, KVTransferConfig

from .base import BaseKVConnector
from .metadata import (
    ConnectorTransferPlan,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


class MultiConnector(BaseKVConnector):
    name = "MultiConnector"

    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
        children: Iterable[BaseKVConnector] | None = None,
    ) -> None:
        super().__init__(config, cache_config)
        self.children = list(children or [])

    def set_simulation_time(self, now: float) -> None:
        super().set_simulation_time(now)
        for child in self.children:
            child.set_simulation_time(now)

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        loads: list[ConnectorTransferPlan] = []
        saves: list[ConnectorTransferPlan] = []
        preempted: list[int] = []
        for child in self.children:
            meta = child.build_connector_meta(scheduler_output)
            loads.extend(meta.loads)
            saves.extend(meta.saves)
            preempted.extend(meta.preempted_request_ids)
        return KVConnectorMetadata(
            connector_name=self.name,
            loads=loads,
            saves=saves,
            preempted_request_ids=preempted,
        )

    def build_transfer_plan(self, **kwargs) -> ConnectorTransferPlan:
        for child in self.children:
            builder = getattr(child, "build_transfer_plan", None)
            if builder is not None:
                return builder(**kwargs)
        raise AttributeError("MultiConnector has no child that can build a transfer plan")

    def on_new_request(self, req) -> None:
        for child in self.children:
            child.on_new_request(req)

    def get_num_new_matched_tokens(self, req, num_computed_tokens: int) -> int:
        return max(
            (
                child.get_num_new_matched_tokens(req, num_computed_tokens)
                for child in self.children
            ),
            default=0,
        )

    def update_state_after_alloc(self, req, blocks, num_external_tokens: int) -> None:
        for child in self.children:
            child.update_state_after_alloc(req, blocks, num_external_tokens)

    def update_connector_output(
        self,
        worker_output: KVConnectorWorkerMetadata,
    ) -> None:
        for child in self.children:
            child.update_connector_output(worker_output)
        super().update_connector_output(worker_output)

    def request_finished(self, req, block_ids):
        delay = False
        payload = {}
        for child in self.children:
            child_delay, child_payload = child.request_finished(req, block_ids)
            delay = delay or child_delay
            if child_payload:
                payload[child.name] = child_payload
        return delay, payload or None

    def on_request_released(self, req) -> None:
        for child in self.children:
            child.on_request_released(req)

    def bind_connector_metadata(self, meta: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(meta)
        for child in self.children:
            child.bind_connector_metadata(meta)

    def start_load_kv(self) -> float:
        return sum(child.start_load_kv() for child in self.children)

    def wait_for_save(self) -> float:
        return sum(child.wait_for_save() for child in self.children)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        meta = super().build_connector_worker_meta()
        for child in self.children:
            meta = meta.aggregate(child.build_connector_worker_meta())
        return meta
