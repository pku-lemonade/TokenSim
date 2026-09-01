from __future__ import annotations

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError

from .base import BaseKVConnector
from .multi import MultiConnector


class KVConnectorFactory:
    _registry: dict[str, type[BaseKVConnector]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        connector_cls: type[BaseKVConnector],
    ) -> None:
        if name in cls._registry:
            raise ConfigurationError(f"connector {name!r} is already registered")
        cls._registry[name] = connector_cls

    @classmethod
    def get_connector_class(cls, name: str) -> type[BaseKVConnector]:
        if name not in cls._registry:
            raise ConfigurationError(
                f"unsupported kv_connector {name!r}; expected one of "
                + f"{sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def create_connector(
        cls,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
    ) -> BaseKVConnector:
        connector_cls = cls.get_connector_class(config.kv_connector or "NoopConnector")
        if connector_cls is MultiConnector:
            children = []
            for child_config in config.kv_connector_extra_config.get("connectors", []):
                child = KVTransferConfig(**child_config)
                children.append(cls.create_connector(child, cache_config))
            return connector_cls(config, cache_config, children=children)
        return connector_cls(config, cache_config)

    @classmethod
    def registered_connectors(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))
