"""Grouped DRAM pages and queued asynchronous transfers for Mooncake storage.

Reference contract: vLLM Mooncake store worker/coordinator at 30f5c01d4ebb.
This models complete retained group pages, readiness and DMA ownership;
it does not run the Mooncake C++ service or a model server.
"""

from dataclasses import dataclass

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.native_storage import NativePageStore, QueuedKVLink
from TokenSim.mooncake.config import parse_mooncake_config
from TokenSim.mooncake.metrics import MooncakeStats
from TokenSim.mooncake.service import shared_store

from .base import BaseKVConnector
from .simple_cpu import AsyncOffloadConnector


@dataclass(frozen=True)
class _PageView:
    # GPU import must see the original group/prefix/subslot identity. Only
    # the external service wraps that key in a model/layout/shard namespace.
    key: tuple
    page: object

    @property
    def bytes(self):
        return self.page.bytes


class _StoreView:
    def __init__(self, backing, namespace):
        self.backing, self.namespace = backing, namespace

    def lookup(self, key):
        page = self.backing.lookup((self.namespace, key))
        return _PageView(key, page) if page is not None else None

    def reserve_write(self, key, size):
        page = self.backing.reserve_write((self.namespace, key), size)
        return _PageView(key, page) if page is not None else None

    def finish_write(self, view):
        self.backing.finish_write(view.page)

    def acquire(self, view):
        self.backing.acquire(view.page)

    def release(self, view):
        self.backing.release(view.page)

    @property
    def used_bytes(self):
        return self.backing.used_bytes

    @property
    def capacity_bytes(self):
        return self.backing.capacity_bytes

    def validate(self):
        return self.backing.validate()

    def describe(self):
        return self.backing.describe()


class _TransferPath:
    """Conservative serial PCIe + shared network path in one direction.

    Each worker has one PCIe queue; all workers contend on the same node
    network queue. The two directions use independent queues. Streaming
    overlap within a single DMA is not modeled.
    """

    def __init__(self, env, host, network, *, read):
        self.env, self.host, self.network, self.read = env, host, network, read

    def transfer(self, size):
        links = (self.network, self.host) if self.read else (self.host, self.network)
        for link in links:
            yield self.env.process(link.transfer(size))

    def describe(self):
        return {
            "path": (
                "shared-network then PCIe" if self.read else "PCIe then shared-network"
            ),
            "host": self.host.describe(),
            "shared_network": self.network.describe(),
        }


class _SharedService:
    def __init__(self, worker, config):
        self.config = config
        self.connectors = []
        self.contributors = {
            id(worker.engine): (worker.engine, len(worker.engine.workers))
        }
        world_size = len(worker.engine.workers)
        self.staging_bytes = config.local_buffer_size * world_size
        contributions = world_size if config.mode == "embedded" else 1
        self.capacity_bytes = config.global_segment_size * contributions
        self.budget_bytes = self.capacity_bytes + self.staging_bytes
        if (
            config.host_available_bytes is not None
            and self.budget_bytes > config.host_available_bytes
        ):
            raise ConfigurationError(
                "Mooncake shared budget exceeds available host DRAM"
            )
        if self.capacity_bytes <= 0:
            raise ConfigurationError("Mooncake store capacity must be positive")
        self.store = NativePageStore(self.capacity_bytes, scope="mooncake:shared-dram")
        if config.mode == "embedded":
            return
        # Cap the aggregate by the configured node network injection
        # link; parallel paths must not replicate node bandwidth per worker.
        from TokenSim.mooncake.transfer_engine import TransferEngineSimulator

        transfer = TransferEngineSimulator(config)
        node_link = worker.hardware_context.link(worker.nettype)
        bandwidth = min(
            transfer._bandwidth(config.protocol)
            * max(config.parallel_paths, config.num_nics),
            node_link.bandwidth_per_direction_bytes_per_s,
        )
        args = {
            "utilization": config.transfer_utilization,
            "latency_us": transfer._fixed_latency(config.protocol, "cross_node") * 1e6,
        }
        self.read = QueuedKVLink(worker.env, bandwidth, **args)
        self.write = QueuedKVLink(worker.env, bandwidth, **args)

    def register(self, worker):
        proposed = {
            **self.contributors,
            id(worker.engine): (worker.engine, len(worker.engine.workers)),
        }
        ranks = sum(item[1] for item in proposed.values())
        capacity = self.config.global_segment_size * (
            ranks if self.config.mode == "embedded" else 1
        )
        staging = self.config.local_buffer_size * ranks
        if (
            self.config.host_available_bytes is not None
            and capacity + staging > self.config.host_available_bytes
        ):
            raise ConfigurationError(
                "shared Mooncake pool and staging exceed available host DRAM"
            )
        self.contributors = proposed
        self.capacity_bytes, self.staging_bytes = capacity, staging
        self.budget_bytes = capacity + staging
        self.store.capacity_bytes = capacity


class PageStore(AsyncOffloadConnector):
    """Grouped-page storage behind the ordinary Mooncake connector interface.

    Reuses the tested request-local async import and DMA release lifecycle.
    Storage, key namespaces, budgets and transfer paths are shared-store
    semantics, rather than SimpleCPU's packed rank-local pool semantics.
    """

    name = "MooncakeStoreConnector"

    def __init__(self, config, cache_config):
        super().__init__(config, cache_config)
        self.store_config = parse_mooncake_config(config.kv_connector_extra_config)
        c = self.store_config
        if c.save_policy != "mooncake":
            raise ConfigurationError(
                "grouped Mooncake storage supports save_policy=mooncake"
            )
        if (
            c.enable_offload
            or c.replica_num != 1
            or c.memory_capacity_blocks is not None
        ):
            raise ConfigurationError(
                "native Mooncake supports single-replica DRAM byte budgets"
            )
        if not (c.load_async and c.transfer_overlap) or c.eviction_policy != "lru":
            raise ConfigurationError(
                "native Mooncake requires asynchronous transfers and LRU"
            )
        if c.protocol not in ("rdma", "tcp"):
            raise ConfigurationError("grouped Mooncake storage supports RDMA/TCP")
        if c.preferred_segment is not None:
            raise ConfigurationError(
                "grouped storage does not model separate owner segments"
            )
        self.mooncake_stats = MooncakeStats()

    def bind_runtime(self, worker):
        if worker.parallel_config.pipeline_parallel_size != 1:
            raise ConfigurationError("native Mooncake currently supports PP1")
        self.worker, self.env = worker, worker.env
        self.manager = worker.scheduler.block_manager
        self.scheduler = worker.scheduler
        service = shared_store(
            worker.env,
            self.store_config,
            ("grouped",),
            lambda: _SharedService(worker, self.store_config),
        )
        service.register(worker)
        if (
            service.connectors
            and service.connectors[0].worker.network != worker.network
        ):
            raise ConfigurationError(
                "native Mooncake currently requires one compute node"
            )
        self.service = service
        service.connectors.append(self)
        groups = tuple(
            (
                g.group_id,
                g.eagle,
                tuple(
                    (
                        p.name,
                        p.kind,
                        p.block_tokens,
                        p.state_bytes,
                        p.tokens_per_state,
                        p.window_tokens,
                        p.alignment,
                        p.draft,
                    )
                    for p in g.layers
                ),
            )
            for g in self.manager.layout.groups
        )
        # DP identity is deliberately absent: compatible replicas share KV.
        # Cache salt and the complete prefix chain are already in each key.
        namespace = (
            self.store_config.tenant_id,
            self.store_config.cache_prefix,
            worker.model_spec.model_id,
            worker.model_spec.dtype,
            tuple(sorted(worker.model_spec.operator_dtypes.items())),
            worker.tp_rank,
            worker.pp_rank,
            groups,
            self.store_config.group_id,
            self.store_config.pcp_rank,
            self.store_config.dcp_rank,
        )
        self.store = _StoreView(service.store, namespace)
        host_link = worker.hardware_context.link(worker.device.host_link)
        host_args = {
            "utilization": self.store_config.transfer_utilization,
            "latency_us": host_link.latency_us,
        }
        self.host_read = QueuedKVLink(
            self.env, host_link.bandwidth_per_direction_bytes_per_s, **host_args
        )
        self.host_write = QueuedKVLink(
            self.env, host_link.bandwidth_per_direction_bytes_per_s, **host_args
        )
        self.read_link = self.host_read
        self.write_link = self.host_write
        if self.store_config.mode == "standalone-store":
            self.read_link = _TransferPath(
                self.env, self.host_read, service.read, read=True
            )
            self.write_link = _TransferPath(
                self.env, self.host_write, service.write, read=False
            )
        self.budget_bytes = service.budget_bytes
        self.manager.publish_callbacks.append(self._queue_write)
        self.changed = self.env.event()

    def probe(self, req, local_plan):
        if self.config.kv_role == "kv_producer":
            return local_plan.hit_tokens, ()
        boundary, hits = super().probe(req, local_plan)
        self.mooncake_stats.get_count += 1
        self.mooncake_stats.cache_query_tokens += max(
            0, req.context_len - local_plan.hit_tokens
        )
        self.mooncake_stats.local_gpu_hit_tokens += local_plan.hit_tokens
        if hits:
            self.mooncake_stats.store_hit_count += 1
            self.mooncake_stats.memory_tier_hit_count += 1
            self.mooncake_stats.mooncake_memory_hit_tokens += max(
                0, boundary - local_plan.hit_tokens
            )
        else:
            self.mooncake_stats.store_miss_count += 1
        return boundary, hits

    def _queue_write(self, group, index, block, req):
        if self.store_config.admission_policy == "always":
            super()._queue_write(group, index, block, req)

    def begin_load(self, req, boundary, hits):
        super().begin_load(req, boundary, hits)

    def _record_transfer(self, **values):
        super()._record_transfer(**values)
        kind = values["kind"]
        if kind == "save":
            self.mooncake_stats.put_count += 1
        self.mooncake_stats.record_transfer(
            bytes_=values["bytes_"],
            latency=values["latency"],
            kind=f"store_{kind}",
            blocking_latency=values.get("blocking_latency", 0.0),
        )

    def _transfer_bytes(self, pages):
        return sum(page.bytes for page in pages)

    def describe(self):
        return {
            "backend": self.name,
            "native_async": True,
            "service_scope": "environment master-shared DRAM; compatible engines and DP replicas",
            "key_scope": "model/layout/TP/PP + group/complete salted prefix/subslot; DP excluded",
            "runtime_budget_shared_bytes": self.service.budget_bytes,
            "staging_bytes": self.service.staging_bytes,
            "storage": self.store.describe(),
            "read": self.read_link.describe(),
            "write": self.write_link.describe(),
            "loading_requests": len(self.loads),
            "write_jobs": self.write_jobs,
            "save_scope": "newly computed complete retained group pages; duplicate writes suppressed",
            "deployment_mode": self.store_config.mode,
            "source_scope_limit": "DRAM only; one compute node; embedded PCIe or standalone serial PCIe/network approximation",
        }
