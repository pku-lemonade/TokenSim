from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from TokenSim.errors import ConfigurationError


WorkerT = TypeVar("WorkerT")


class PlacementPolicy(Protocol[WorkerT]):
    def schedule(self) -> WorkerT:
        ...


class WorkerPool(Generic[WorkerT]):
    def __init__(self, workers: list[WorkerT]):
        if not workers:
            raise ConfigurationError("worker pool requires at least one worker")
        self.workers = workers
        self.start_idx = workers[0].id
        self.placement_decision_count = 0

    def schedule(self) -> WorkerT:
        raise NotImplementedError

    def select_prefill_worker(self, request=None) -> WorkerT:
        return self._select()

    def select_decode_worker(self, request=None) -> WorkerT:
        return self._select()

    def select_transfer_target(self, requests=None) -> WorkerT:
        return self._select()

    def _select(self) -> WorkerT:
        self.placement_decision_count += 1
        return self.schedule()

    def get(self, id: int) -> WorkerT:
        return self.workers[id - self.start_idx]

    def __len__(self) -> int:
        return len(self.workers)


class DataParallelWorkerPool(WorkerPool[WorkerT]):
    def __init__(
        self,
        workers: list[WorkerT],
        pool_cls: type[WorkerPool[WorkerT]],
    ):
        super().__init__(workers)
        grouped: dict[int, list[WorkerT]] = {}
        for worker in workers:
            grouped.setdefault(getattr(worker, "dp_rank", 0), []).append(worker)
        self.group_pools = {
            dp_rank: pool_cls(group_workers)
            for dp_rank, group_workers in sorted(grouped.items())
        }
        self.dp_ranks = sorted(self.group_pools)
        self.cur_dp_idx = 0
        self.dp_placement_counts = {rank: 0 for rank in self.dp_ranks}

    def schedule(self) -> WorkerT:
        return self._select()

    def select_prefill_worker(self, request=None) -> WorkerT:
        worker = self._select_for_request(request, assign_new=True)
        self.placement_decision_count += 1
        return worker

    def select_decode_worker(self, request=None) -> WorkerT:
        worker = self._select_for_request(request, assign_new=False)
        self.placement_decision_count += 1
        return worker

    def select_transfer_target(self, requests=None) -> WorkerT:
        request = requests[0] if requests else None
        worker = self._select_for_request(request, assign_new=False)
        self.placement_decision_count += 1
        return worker

    def _select(self) -> WorkerT:
        worker = self._next_group_pool().schedule()
        self._record(worker)
        return worker

    def _select_for_request(self, request=None, assign_new: bool = False) -> WorkerT:
        dp_rank = getattr(request, "dp_rank", None)
        if dp_rank in self.group_pools:
            worker = self.group_pools[dp_rank].schedule()
        else:
            worker = self._next_group_pool().schedule()
        self._record(worker)
        if request is not None and (assign_new or getattr(request, "dp_rank", None) is None):
            setattr(request, "dp_rank", getattr(worker, "dp_rank", 0))
        return worker

    def _next_group_pool(self) -> WorkerPool[WorkerT]:
        dp_rank = self.dp_ranks[self.cur_dp_idx]
        self.cur_dp_idx = (self.cur_dp_idx + 1) % len(self.dp_ranks)
        return self.group_pools[dp_rank]

    def _record(self, worker: WorkerT) -> None:
        dp_rank = getattr(worker, "dp_rank", 0)
        self.dp_placement_counts[dp_rank] = self.dp_placement_counts.get(dp_rank, 0) + 1


class RoundRobinWorkerPool(WorkerPool[WorkerT]):
    def __init__(self, workers: list[WorkerT]):
        super().__init__(workers)
        self.cur_idx = 0
        self.num_workers = len(workers)

    def schedule(self) -> WorkerT:
        i = self.cur_idx
        self.cur_idx = (self.cur_idx + 1) % self.num_workers
        return self.workers[i]


class LeastGpuMemoryWorkerPool(WorkerPool[WorkerT]):
    def workloads(self) -> list[float]:
        return [worker.workload() for worker in self.workers]

    def schedule(self) -> WorkerT:
        workloads = self.workloads()
        i = workloads.index(min(workloads))
        return self.workers[i]


@dataclass(frozen=True, order=True)
class WorkerLoad:
    active_requests: int
    gpu_memory_percent: float
    cpu_transfer_gb: float
    worker_id: int


class BalancedLoadWorkerPool(WorkerPool[WorkerT]):
    def worker_load(self, worker: WorkerT) -> WorkerLoad:
        scheduler = worker.scheduler
        active_requests = (
            len(getattr(scheduler, "running", []))
            + len(getattr(scheduler, "waiting", []))
        )
        return WorkerLoad(
            active_requests=active_requests,
            gpu_memory_percent=worker.workload(),
            cpu_transfer_gb=worker.cpu_workload(),
            worker_id=worker.id,
        )

    def schedule(self) -> WorkerT:
        return min(
            self.workers,
            key=lambda worker: self.worker_load(worker),
        )
