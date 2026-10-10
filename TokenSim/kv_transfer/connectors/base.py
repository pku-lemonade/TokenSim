from __future__ import annotations

from typing import Any

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.llm.llm_request import Request

from .metadata import ConnectorStats, KVConnectorMetadata, KVConnectorWorkerMetadata


class BaseKVConnector:
    name = "BaseKVConnector"
    native_async = False
    supports_grouped_cache = False

    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
    ) -> None:
        self.config = config
        self.cache_config = cache_config
        self.stats = ConnectorStats()
        self.stats.record_role(config.kv_role)
        self._metadata = KVConnectorMetadata(connector_name=self.name)
        self.simulation_time = 0.0

    def set_simulation_time(self, now: float) -> None:
        self.simulation_time = float(now)

    def bind_runtime(self, worker) -> None:
        return

    def on_new_request(self, req: Request) -> None:
        return

    def get_num_new_matched_tokens(
        self,
        req: Request,
        num_computed_tokens: int,
    ) -> int:
        return 0

    def update_state_after_alloc(
        self,
        req: Request,
        blocks: list[Any],
        num_external_tokens: int,
    ) -> None:
        return

    def discard_load(self, req: Request) -> None:
        """Discard a cache probe that was not selected for this admission."""
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return KVConnectorMetadata(connector_name=self.name)

    def update_connector_output(
        self,
        worker_output: KVConnectorWorkerMetadata,
    ) -> None:
        return

    def request_finished(
        self,
        req: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def on_request_released(self, req: Request) -> None:
        """Notify that a request left this engine instance.

        Mirrors the reference connector dropping per-request scheduler state
        for ``finished_req_ids``; also invoked when a prefill worker hands a
        request over to a decode worker.
        """
        return

    def bind_connector_metadata(self, meta: KVConnectorMetadata) -> None:
        self._metadata = meta

    def handle_preemptions(self, meta: KVConnectorMetadata) -> None:
        return

    def start_load_kv(self) -> float:
        latency = sum(plan.latency for plan in self._metadata.loads)
        for plan in self._metadata.loads:
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=plan.latency,
                kind="load",
            )
        return latency

    def wait_for_layer_load(self, layer: int | str) -> float:
        return 0.0

    def save_kv_layer(self, layer: int | str, requests: list[Request]) -> float:
        return 0.0

    def wait_for_save(self) -> float:
        latency = sum(plan.latency for plan in self._metadata.saves)
        for plan in self._metadata.saves:
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=plan.latency,
                kind="save",
            )
        return latency

    def get_finished(
        self,
        finished_req_ids: set[int],
    ) -> tuple[set[int], set[int]]:
        return set(finished_req_ids), set(finished_req_ids)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        return KVConnectorWorkerMetadata()


def flatten_connectors(connector):
    """Visit a composed connector and its children in declaration order."""
    if connector is None:
        return []
    result = [connector]
    for child in getattr(connector, "children", ()):
        result.extend(flatten_connectors(child))
    return result
