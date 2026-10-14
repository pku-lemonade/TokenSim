from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.errors import ConfigurationError
from TokenSim.storage import StorageBackendFactory


class MooncakeService:
    def __init__(self, config: MooncakeConfig, block_bytes: int):
        self.config = config
        self.block_bytes = block_bytes
        self.store = MooncakeStore(config, block_bytes)
        self.contributors = {}

    def register(self, contributor, ranks, config):
        proposed = {**self.contributors, id(contributor): (contributor, ranks, config)}
        capacity = (
            self.config.memory_capacity_blocks * self.block_bytes
            if self.config.memory_capacity_blocks is not None
            else (
                sum(
                    count * options.global_segment_size
                    for _, count, options in proposed.values()
                )
                if self.config.mode == "embedded"
                else self.config.global_segment_size
            )
        )
        reserved = capacity + sum(
            count * options.local_buffer_size for _, count, options in proposed.values()
        )
        if (
            self.config.host_available_bytes is not None
            and reserved > self.config.host_available_bytes
        ):
            raise ConfigurationError(
                "shared Mooncake pool and staging exceed available host DRAM"
            )
        self.contributors = proposed
        self.store.memory_capacity_bytes = capacity


def shared_store(owner, config, payload_format, create):
    """One simulated master per environment, independent of engine identity."""
    if owner is None:
        return create()
    stores = getattr(owner, "mooncake_stores", None)
    if stores is None:
        stores = owner.mooncake_stores = {}
    identity = config.master_server_address
    signature = _config_fingerprint(config)
    if identity not in stores:
        stores[identity] = (signature, payload_format, create())
    expected, payload, service = stores[identity]
    if signature != expected or payload != payload_format:
        raise ConfigurationError(
            "engines sharing a Mooncake master need compatible store settings and payload formats"
        )
    return service


def get_mooncake_service(
    kv_config: KVTransferConfig,
    *,
    block_bytes: int,
    extra_config: dict[str, Any] | None = None,
    owner=None,
    contributor=None,
    ranks=1,
) -> MooncakeService:
    mooncake_config = parse_mooncake_config(
        extra_config
        if extra_config is not None
        else kv_config.kv_connector_extra_config
    )

    def create():
        service = MooncakeService(mooncake_config, block_bytes)
        service.register(
            contributor if contributor is not None else service, ranks, mooncake_config
        )
        return service

    service = shared_store(
        owner,
        mooncake_config,
        ("uniform", block_bytes),
        create,
    )
    service.register(
        contributor if contributor is not None else service, ranks, mooncake_config
    )
    return service


def _config_fingerprint(config: MooncakeConfig) -> str:
    payload = asdict(config)
    if config.offload is not None:
        payload["offload"]["config"] = StorageBackendFactory.create(
            config.offload
        ).describe()
    client_fields = [
        "tenant_id",
        "cache_prefix",
        "group_id",
        "pcp_rank",
        "dcp_rank",
        "admission_policy",
        "save_policy",
        "load_async",
        "transfer_overlap",
        "local_buffer_size",
    ]
    if config.mode == "embedded":
        client_fields.append("global_segment_size")
    if config.cache_layout == "uniform" or config.mode == "embedded":
        client_fields.extend(
            [
                "protocol",
                "num_nics",
                "parallel_paths",
                "fixed_latency_us",
                "bandwidth_bytes_per_s",
                "transfer_utilization",
                "protocol_bandwidth_bytes_per_s",
                "protocol_latency_us",
            ]
        )
    for name in client_fields:
        payload.pop(name)
    return json.dumps(payload, sort_keys=True, default=str)
