from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import logging


from TokenSim.llm.llm_comm import KVConnectorMetadata, KVConnectorWorkerMetadata
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.block.block_manager import BlockManager
from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.kv_transfer import NoopConnector

logger = logging.getLogger(__name__)


class SchedulePhase(Enum):
    IDLE = auto()
    PREFILL = auto()
    DECODE = auto()
    RECOMPUTE = auto()


@dataclass
class ScheduleOutput:
    running: list[Request]
    preempted: list[Request]
    scheduled: list[Request]
    connector_metadata: KVConnectorMetadata
    phase: SchedulePhase = SchedulePhase.IDLE


class LLMScheduler(ABC):
    def __init__(self, connector=None):
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.connector = connector or NoopConnector(KVTransferConfig.default())

    def add_requests(self, requests: list[Request]):
        for req in requests:
            self.connector.on_new_request(req)
        self.waiting.extend(requests)

    @abstractmethod
    def schedule(self) -> tuple[list[Request], list[Request]]:
        raise NotImplementedError

    def schedule_output(self) -> ScheduleOutput:
        running, preempted = self.schedule()
        phase = SchedulePhase.IDLE
        if running:
            phase = (
                SchedulePhase.PREFILL if running[0].is_prefill else SchedulePhase.DECODE
            )
        output = ScheduleOutput(
            running=running,
            preempted=preempted or [],
            scheduled=running if not preempted else [],
            connector_metadata=KVConnectorMetadata(),
            phase=phase,
        )
        output.connector_metadata = self.connector.build_connector_meta(output)
        return output

    def update(self, requests: list[Request]):
        # update block manager
        for req in requests:
            if req.is_done:
                self.free(req)
        # update scheduler running
        self.running = [req for req in self.running if not req.is_done]
        return self.running

    def update_connector_output(self, worker_output: KVConnectorWorkerMetadata) -> None:
        self.connector.update_connector_output(worker_output)

    def free(self, req: Request):
        req.status = RequestStatus.FINISHED_STOPPED

    def workload(self):
        return 0


class LLMDynamicScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=200, connector=None):
        super().__init__(connector=connector)
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request]]:
        running = []
        while self.waiting:
            request = self.waiting.pop()
            running.append(request)
            if sum([req.prefill_len for req in running]) == self.max_parallem_sum:
                break
        if running:
            self.running.extend(running)
            return self.running, []

        self.running = sorted(self.running, key=lambda r: r.arrival_time)
        while self.running:
            running.append(self.running.pop())
        self.running.extend(running)
        return self.running, []

    def cpu_workload(self):
        return 0


class LLMStaticScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=0, connector=None):
        super().__init__(connector=connector)
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request]]:
        running = []
        while self.running:
            running.append(self.running.pop())
            if sum([req.prefill_len for req in running]) == self.max_parallem_sum:
                break
        if running:
            self.running.extend(running)
            return running, []
        while self.waiting:
            running.append(self.waiting.pop())
            if sum([req.prefill_len for req in running]) == self.max_parallem_sum:
                break
        self.running.extend(running)
        return running, []

    def cpu_workload(self):
        return 0


class LLMPrefillScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=0, connector=None):
        super().__init__(connector=connector)
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request]]:
        running = []
        if len(self.waiting) >= self.max_parallem_sum:
            while self.waiting:
                running.append(self.waiting.pop())
        else:
            if self.running:
                while self.running:
                    running.append(self.running.pop())
            else:
                while self.waiting:
                    running.append(self.waiting.pop())
        self.running.extend(running)
        return running, []


class LLMPagedAttnScheduler(LLMScheduler):
    def __init__(
        self,
        id: int,
        cache_config: CacheConfig,
        max_parallem_sum=None,
        max_occupy_ratio: float = 1,
        connector=None,
    ):
        super().__init__(connector=connector)

        self.id = id
        self.cache_config = cache_config
        self.max_parallem_sum = max_parallem_sum

        self.block_manager: BlockManager = BlockManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            model=self.cache_config.model,
        )

        self.max_occupy_ratio = max_occupy_ratio
        set_releaser = getattr(self.connector, "set_block_releaser", None)
        if set_releaser is not None:
            set_releaser(self._release_delayed_blocks)

    def schedule(self) -> tuple[list[Request], list[Request]]:
        """The scheduler FSM for dynamic scheduling, basically being ported
            from the vLLM's Scheduler.schedule() method.

            currently we donnot consider requests whose decode length exceeds
            the max_len limitation.

        Return:
            A list of candidate requests for the next decode step.
        """
        scheduled: list[Request] = []
        admission_phase: SchedulePhase | None = None
        while self.waiting:
            if not self._is_occupy_below_usage():
                break
            if (
                self.max_parallem_sum is not None
                and len(self.running) >= self.max_parallem_sum
            ):
                break

            req = self.waiting[0]
            req_phase = (
                SchedulePhase.RECOMPUTE
                if req.needs_recompute
                else SchedulePhase.PREFILL
            )
            if admission_phase is not None and req_phase != admission_phase:
                break

            if not self.block_manager.can_allocate(req):
                break

            req = self.waiting.pop(0)
            local_plan = self.block_manager.kv_cache_manager.plan_reuse(req)
            num_external_tokens = self.connector.get_num_new_matched_tokens(
                req,
                local_plan.hit_tokens,
            )
            if req.needs_recompute:
                req.recompute_tokens = max(
                    0,
                    req.context_len - local_plan.hit_tokens - num_external_tokens,
                )
            self.block_manager.allocate(req)
            self.connector.update_state_after_alloc(
                req,
                self.block_manager.block_table.get_blocks(req.id),
                num_external_tokens,
            )
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            scheduled.append(req)
            admission_phase = req_phase

            # if sum([req.prefill_len for req in scheduled]) >= self.max_parallem_sum:
            #     break

        if scheduled:
            return scheduled, []

        # [TODO: xuechao] sort self.running according to some priority policy.
        self.running = sorted(self.running, key=lambda req: req.arrival_time)

        # Reserve new token slots for the running sequence groups.
        running: list[Request] = []
        preempted: list[Request] = []

        while self.running:
            req = self.running.pop(0)
            while True:
                if self.block_manager.can_append_slot(req):
                    # Append new slots to the sequence group.
                    self.block_manager.append_slot(req)
                    running.append(req)
                    break
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_req = self.running.pop(-1)
                    self._preempt(victim_req)
                    preempted.append(victim_req)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(req)
                    preempted.append(req)
                    break
        self.running = running

        return self.running, preempted

    def schedule_output(self) -> ScheduleOutput:
        running, preempted = self.schedule()
        phase = SchedulePhase.IDLE
        if running:
            if running[0].needs_recompute:
                phase = SchedulePhase.RECOMPUTE
            elif running[0].is_prefill:
                phase = SchedulePhase.PREFILL
            else:
                phase = SchedulePhase.DECODE
        output = ScheduleOutput(
            running=running,
            preempted=preempted or [],
            scheduled=running if not preempted else [],
            connector_metadata=KVConnectorMetadata(
                preempted_request_ids=[req.id for req in preempted or []],
            ),
            phase=phase,
        )
        output.connector_metadata = self.connector.build_connector_meta(output)
        if preempted and not output.connector_metadata.preempted_request_ids:
            output.connector_metadata.preempted_request_ids = [
                req.id for req in preempted
            ]
        return output

    def update(self, requests: list[Request]):
        for req in requests:
            if req.generation_idx > 0:
                self.block_manager.commit_input_cache(req)
        return super().update(requests)

    def update_connector_output(self, worker_output: KVConnectorWorkerMetadata) -> None:
        super().update_connector_output(worker_output)
        if worker_output.finished_sending:
            self.running = [
                req
                for req in self.running
                if req.id not in worker_output.finished_sending
            ]

    def _is_occupy_below_usage(self):
        (_, used_blks, all_blks) = self.block_manager.get_gpu_status()
        return used_blks / all_blks < self.max_occupy_ratio

    def _preempt(self, req: Request) -> None:
        if req.status != RequestStatus.RUNNING:
            logger.debug("preempting request %s with status %s", req.id, req.status)
        self.block_manager.release_request_blocks(req)
        req.prepare_recompute()
        req.status = RequestStatus.WAITING
        self.waiting.insert(0, req)

    def finish_recompute(self, requests: list[Request], latency: float) -> None:
        for req in requests:
            self.block_manager.commit_input_cache(req)
            req.finish_recompute(latency)

    def release_reserved_blocks(self, num_blocks: int | None = None):
        self.block_manager.release_reserved_blocks(num_blocks)

    def free(self, req: Request):
        block_ids = [
            block.block_number
            for block in self.block_manager.block_table.get_blocks(req.id)
        ]
        delay_free, _ = self.connector.request_finished(req, block_ids)
        if delay_free:
            req.status = RequestStatus.WAITING_FOR_CONNECTOR_FREE
            return
        self.block_manager.free(req)
        req.status = RequestStatus.FINISHED_STOPPED

    def workload(self):
        """Return GPU KV-cache utilization as a percentage."""
        free_blocks, used_blocks, total_blocks = self.block_manager.get_gpu_status()
        return (used_blocks / total_blocks) * 100 if total_blocks > 0 else 0

    def cpu_workload(self):
        """Return allocated CPU KV-cache capacity in GiB."""
        _GB = 1 << 30
        capacity = (
            self.block_manager.cpu_allocator.get_num_allocated_blocks()
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
        )
        # return self.block_manager.cpu_allocator.get_num_allocated_blocks()

        return capacity

    def _release_delayed_blocks(self, req: Request, block_ids: list[int]) -> None:
        self.block_manager.release_request_blocks(req)
        req.status = RequestStatus.FINISHED_STOPPED
