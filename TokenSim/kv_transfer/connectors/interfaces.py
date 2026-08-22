from __future__ import annotations

from typing import Any, Protocol

from TokenSim.llm.llm_request import Request

from .metadata import ConnectorStats, KVConnectorMetadata, KVConnectorWorkerMetadata


class SchedulerSideConnector(Protocol):
    stats: ConnectorStats

    def on_new_request(self, req: Request) -> None:
        ...

    def get_num_new_matched_tokens(
        self,
        req: Request,
        num_computed_tokens: int,
    ) -> int:
        ...

    def update_state_after_alloc(
        self,
        req: Request,
        blocks: list[Any],
        num_external_tokens: int,
    ) -> None:
        ...

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        ...

    def update_connector_output(
        self,
        worker_output: KVConnectorWorkerMetadata,
    ) -> None:
        ...

    def request_finished(
        self,
        req: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        ...

    def take_events(self) -> list[str]:
        ...

    def reset_cache(self) -> bool:
        ...


class WorkerSideConnector(Protocol):
    stats: ConnectorStats

    def bind_connector_metadata(self, meta: KVConnectorMetadata) -> None:
        ...

    def handle_preemptions(self, meta: KVConnectorMetadata) -> None:
        ...

    def start_load_kv(self) -> float:
        ...

    def wait_for_layer_load(self, layer: int | str) -> float:
        ...

    def save_kv_layer(self, layer: int | str, requests: list[Request]) -> float:
        ...

    def wait_for_save(self) -> float:
        ...

    def get_finished(
        self,
        finished_req_ids: set[int],
    ) -> tuple[set[int], set[int]]:
        ...

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        ...
