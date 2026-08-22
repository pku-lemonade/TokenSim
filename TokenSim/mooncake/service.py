from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.mooncake.transfer_engine import TransferEngineSimulator


class MooncakeService:
    def __init__(self, config: MooncakeConfig, block_bytes: int):
        self.config = config
        self.block_bytes = block_bytes
        self.store = MooncakeStore(config, block_bytes)
        self.transfer_engine = TransferEngineSimulator(config)


_SERVICES: dict[tuple[str, str], MooncakeService] = {}


def get_mooncake_service(
    kv_config: KVTransferConfig,
    *,
    block_bytes: int,
    extra_config: dict[str, Any] | None = None,
) -> MooncakeService:
    mooncake_config = parse_mooncake_config(
        extra_config if extra_config is not None else kv_config.kv_connector_extra_config
    )
    key = (
        kv_config.engine_id or "default",
        _config_fingerprint(mooncake_config, block_bytes),
    )
    if key not in _SERVICES:
        _SERVICES[key] = MooncakeService(mooncake_config, block_bytes)
    return _SERVICES[key]


def reset_mooncake_services() -> None:
    _SERVICES.clear()


def _config_fingerprint(config: MooncakeConfig, block_bytes: int) -> str:
    payload = asdict(config)
    payload["block_bytes"] = block_bytes
    return json.dumps(payload, sort_keys=True, default=str)
