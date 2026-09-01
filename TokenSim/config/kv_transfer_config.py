import json
import uuid
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass, Field

from TokenSim.config.constants import KV_TRANSFER_ROLES, WORKER_ROLES
from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import validate_mooncake_connector_config


@dataclass
class KVTransferConfig:
    kv_connector: str | None = None
    kv_role: str | None = None
    kv_buffer_device: str = "cuda"
    kv_connector_extra_config: dict[str, Any] = Field(default_factory=dict)
    kv_parallel_size: int = 1
    kv_rank: int | None = None
    engine_id: str | None = None

    def __post_init__(self) -> None:
        if self.kv_connector in (None, ""):
            self.kv_connector = "NoopConnector"
        if self.engine_id in (None, ""):
            self.engine_id = str(uuid.uuid4())
        if self.kv_role is not None and self.kv_role not in KV_TRANSFER_ROLES:
            raise ConfigurationError(
                f"unsupported kv_role {self.kv_role!r}; expected one of "
                + f"{sorted(KV_TRANSFER_ROLES)}"
            )
        if self.kv_parallel_size < 1:
            raise ConfigurationError("kv_parallel_size must be at least 1")
        validate_mooncake_connector_config(
            self.kv_connector,
            self.kv_connector_extra_config,
        )

    @classmethod
    def from_file(cls, filename: str | Path) -> "KVTransferConfig":
        return cls(**json.loads(Path(filename).read_text()))

    @classmethod
    def default(cls) -> "KVTransferConfig":
        return cls(kv_connector="NoopConnector")

    def with_worker_defaults(
        self,
        worker_role: str,
        kv_rank: int | None = None,
    ) -> "KVTransferConfig":
        role = self.kv_role or infer_kv_role(worker_role)
        return KVTransferConfig(
            kv_connector=self.kv_connector,
            kv_role=role,
            kv_buffer_device=self.kv_buffer_device,
            kv_connector_extra_config=dict(self.kv_connector_extra_config),
            kv_parallel_size=self.kv_parallel_size,
            kv_rank=self.kv_rank if kv_rank is None else kv_rank,
            engine_id=self.engine_id,
        )


def infer_kv_role(worker_role: str) -> str:
    if worker_role == "prefill":
        return "kv_producer"
    if worker_role == "decode":
        return "kv_consumer"
    if worker_role == "hybrid":
        return "kv_both"
    raise ConfigurationError(
        f"unsupported worker role {worker_role!r}; expected one of "
        + f"{sorted(WORKER_ROLES)}"
    )
