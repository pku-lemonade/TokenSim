"""One Mooncake connector interface with explicit cache-layout configuration."""

from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import parse_mooncake_config

from .store_blocks import BlockStore
from .store_pages import PageStore


class MooncakeStoreConnector:
    name = "MooncakeStoreConnector"
    supports_grouped_cache = True

    def __init__(self, config, cache_config=None):
        options = parse_mooncake_config(config.kv_connector_extra_config)
        grouped = getattr(cache_config, "hybrid_layout", None) is not None
        if (options.cache_layout == "grouped") != grouped:
            raise ConfigurationError(
                "Mooncake cache_layout must match the KV cache: "
                "grouped KV requires cache_layout=grouped; uniform KV requires uniform"
            )
        storage = PageStore if options.cache_layout == "grouped" else BlockStore
        self._storage = storage(config, cache_config)

    def __getattr__(self, name):
        return getattr(self._storage, name)
