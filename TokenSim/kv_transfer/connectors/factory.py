from __future__ import annotations

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError

from .base import BaseKVConnector
from .multi import MultiConnector
from .mooncake_store import MooncakeStoreConnector


class KVConnectorFactory:
    _registry: dict[str, type[BaseKVConnector | MooncakeStoreConnector]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        connector_cls: type[BaseKVConnector | MooncakeStoreConnector],
    ) -> None:
        if name in cls._registry:
            raise ConfigurationError(f"connector {name!r} is already registered")
        cls._registry[name] = connector_cls

    @classmethod
    def get_connector_class(
        cls, name: str
    ) -> type[BaseKVConnector | MooncakeStoreConnector]:
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
    ) -> BaseKVConnector | MooncakeStoreConnector:
        connector_cls = cls.get_connector_class(config.kv_connector or "NoopConnector")
        if connector_cls is MultiConnector:
            children = []
            for child_config in config.kv_connector_extra_config.get("connectors", []):
                effective = dict(child_config)
                for name in ("kv_role", "kv_rank", "engine_id"):
                    if effective.get(name) is None:
                        effective[name] = getattr(config, name)
                child = KVTransferConfig.from_mapping(effective)
                children.append(cls.create_connector(child, cache_config))
            if any(child.async_offload for child in children):
                raise ConfigurationError(
                    "queued storage must be configured as a direct connector; "
                    "MultiConnector does not coordinate asynchronous page ownership"
                )
            return connector_cls(config, cache_config, children=children)
        if (
            getattr(cache_config, "hybrid_layout", None) is not None
            and not connector_cls.supports_grouped_cache
        ):
            raise ConfigurationError(
                f"{config.kv_connector} does not support grouped KV pages"
            )
        return connector_cls(config, cache_config)

    @classmethod
    def registered_connectors(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))
