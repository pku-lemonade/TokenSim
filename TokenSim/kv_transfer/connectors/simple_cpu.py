"""Native SimpleCPU: confirmed eager stores and request-local asynchronous loads.

Source contract: vllm/v1/simple_kv_offload at 30f5c01d4ebb. Each rank has
its own packed CPU pool and full-duplex PCIe queues. DMA references outlive
request references; cancelled loads never publish their destination pages.
"""

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.native_storage import NativePageStore, QueuedKVLink
from TokenSim.llm.llm_request import RequestStatus

from .base import BaseKVConnector


class SimpleCPUOffloadConnector(BaseKVConnector):
    name = "SimpleCPUOffloadConnector"
    native_async = True

    def __init__(self, config, cache_config=None):
        super().__init__(config, cache_config)
        if cache_config is None or cache_config.hybrid_layout is None:
            raise ConfigurationError("SimpleCPU requires native hybrid KV layout")
        extra = config.kv_connector_extra_config
        if extra.get("lazy_offload", False):
            raise ConfigurationError("native SimpleCPU currently supports eager stores")
        if "memory_capacity_blocks" in extra:
            raise ConfigurationError(
                "CPU capacity must be a byte budget, not logged blocks"
            )
        self.extra = extra
        self.loads = {}
        self.pending_writes = []
        self.write_jobs = 0

    def bind_runtime(self, worker):
        self.worker, self.env = worker, worker.env
        self.manager = worker.scheduler.block_manager
        self.scheduler = worker.scheduler
        world_size = worker.parallel_config.world_size
        capacity = self.extra.get("cpu_bytes_to_use_per_rank")
        if capacity is None:
            capacity = self.extra.get("cpu_bytes_to_use", 0) // world_size
        host_bytes = self.extra.get("host_available_bytes")
        if host_bytes is not None and capacity * world_size > host_bytes:
            raise ConfigurationError("CPU runtime budget exceeds available host DRAM")
        stride = self.manager.layout.physical_slot_bytes
        # Both source pools reserve block zero as the null page.
        slots = int(capacity) // stride
        if slots < 2:
            raise ConfigurationError("SimpleCPU needs a null slot and usable CPU slots")
        self.budget_bytes = int(capacity)
        self.store = NativePageStore(
            (slots - 1) * stride, allocation_unit=stride, scope=f"worker:{worker.id}"
        )
        link = worker.hardware_context.link(worker.device.host_link)
        utilization = self.extra.get("transfer_utilization", 0.75)
        args = dict(utilization=utilization, latency_us=link.latency_us)
        self.read_link = QueuedKVLink(
            self.env, link.bandwidth_per_direction_bytes_per_s, **args
        )
        self.write_link = QueuedKVLink(
            self.env, link.bandwidth_per_direction_bytes_per_s, **args
        )
        self.manager.publish_callbacks.append(self._queue_write)
        self.changed = self.env.event()

    def notify(self):
        if not self.changed.triggered:
            self.changed.succeed()
        self.changed = self.env.event()
        coordinator = getattr(self.worker.engine, "dp_coordinator", None)
        if coordinator is not None:
            coordinator.notify()
        elif (
            self.scheduler.running
            or self.scheduler.waiting
            or self.scheduler.loaded_waiting
        ):
            from TokenSim.llm.llm_engine import Task, WorkerStatus

            if self.worker.status == WorkerStatus.WAITING:
                self.worker.status = WorkerStatus.RUNNING
                self.worker.send_task(self.worker, Task.STEP)

    def probe(self, req, local_plan):
        return self.manager.plan_external_reuse(
            req, self.store, local_tokens=local_plan.hit_tokens
        )

    def begin_load(self, req, boundary, hits):
        pages = [p for _, pairs in hits for _, p in pairs]
        for page in pages:
            self.store.acquire(page)
        try:
            targets = self.manager.allocate_external_destinations(req, hits)
        except Exception:
            for page in pages:
                self.store.release(page)
            raise
        for _, _, _, block in targets:
            self.manager.pin_transfer(block)
        req.external_cached_prefill_tokens = boundary - req.gpu_cached_prefill_tokens
        req.cached_prefill_tokens = boundary
        req.cached_prefill_blocks = boundary // req.block_size
        req.effective_prefill_tokens = req.prefill_len - boundary
        req.status = RequestStatus.WAITING_FOR_KV
        self.scheduler.loading[req.id] = req
        self.loads[req.id] = self.env.process(self._load(req, targets, boundary))

    def _load(self, req, targets, boundary):
        issued = self.env.now
        bytes_ = len(targets) * self.manager.layout.physical_slot_bytes
        yield self.env.process(self.read_link.transfer(bytes_))
        # Cancellation releases request references immediately. DMA pins keep
        # targets private and resident until this point, then release them.
        if not req.cancel_requested and req.id in self.manager.tables:
            self.manager.finish_external_import(req, targets, boundary)
            if req.needs_recompute:
                req.recompute_tokens = max(0, req.context_len - boundary)
            req.start_context_build(boundary)
            # Copy completion makes the KV usable, not the request runnable.
            # scheduler._handle_blocked_waiting_request returns to WAITING;
            # the next waiting admission must allocate its actual query slots.
            req.status = RequestStatus.WAITING
            self.scheduler.loaded_waiting.append(req)
        for _, _, page, block in targets:
            self.store.release(page)
            self.manager.unpin_transfer(block)
        if req.cancel_requested:
            self.scheduler.loading.pop(req.id, None)
        self.loads.pop(req.id, None)
        self.stats.record_transfer(
            blocks=len(targets),
            bytes_=bytes_,
            latency=self.env.now - issued,
            kind="load",
        )
        self.notify()

    def _queue_write(self, group, index, block, req):
        page = self.store.reserve_write(block.block_hash, group.group_bytes)
        if page is None:
            return
        self.manager.pin_transfer(block)
        self.pending_writes.append((page, block))

    def wait_for_save(self):
        if self.pending_writes:
            batch, self.pending_writes = self.pending_writes, []
            self.write_jobs += 1
            self.env.process(self._save(batch))
        return 0.0

    def _save(self, batch):
        issued = self.env.now
        bytes_ = len(batch) * self.manager.layout.physical_slot_bytes
        yield self.env.process(self.write_link.transfer(bytes_))
        for page, block in batch:
            self.store.finish_write(page)
            self.manager.unpin_transfer(block)
        self.write_jobs -= 1
        self.stats.record_transfer(
            blocks=len(batch),
            bytes_=bytes_,
            latency=self.env.now - issued,
            kind="save",
            blocking_latency=0.0,
        )
        self.notify()

    def request_finished(self, req, block_ids):
        self.scheduler.loading.pop(req.id, None)
        return False, None

    def reset_cache(self):
        if self.loads or self.write_jobs or self.pending_writes:
            return False
        self.store = NativePageStore(
            self.store.capacity_bytes,
            allocation_unit=self.store.allocation_unit,
            scope=self.store.scope,
        )
        return True

    def describe(self):
        return {
            "backend": self.name,
            "runtime_budget_per_rank_bytes": self.budget_bytes,
            "storage": self.store.describe(),
            "read": self.read_link.describe(),
            "write": self.write_link.describe(),
            "loading_requests": len(self.loads),
            "write_jobs": self.write_jobs,
        }
