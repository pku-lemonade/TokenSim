from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import logging


from TokenSim.llm.llm_comm import KVConnectorMetadata, KVConnectorWorkerMetadata
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.block.block_manager import BlockManager
from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import NoopConnector

logger = logging.getLogger(__name__)

# Token budget of one paged-attention step (vLLM V1 ``max_num_batched_tokens``
# for online serving; SGLang's ``chunked_prefill_size`` on large GPUs uses the
# same value). Prompts longer than the budget are prefilled over several steps.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


class SchedulePhase(Enum):
    IDLE = auto()
    PREFILL = auto()
    DECODE = auto()
    RECOMPUTE = auto()
    # Chunked prefill: decode tokens and prefill/recompute chunks in one step.
    MIXED = auto()


@dataclass
class ScheduleOutput:
    running: list[Request]
    preempted: list[Request]
    # Requests admitted (or re-admitted after preemption) in this step. KV
    # connectors plan their loads and saves for these.
    scheduled: list[Request]
    connector_metadata: KVConnectorMetadata
    phase: SchedulePhase = SchedulePhase.IDLE


def step_phase(requests: list[Request]) -> SchedulePhase:
    """Phase label of a step: one of the pure phases or MIXED."""
    kinds = set()
    for req in requests:
        if req.needs_recompute:
            kinds.add(SchedulePhase.RECOMPUTE)
        elif req.is_prefill:
            kinds.add(SchedulePhase.PREFILL)
        else:
            kinds.add(SchedulePhase.DECODE)
    if not kinds:
        return SchedulePhase.IDLE
    return kinds.pop() if len(kinds) == 1 else SchedulePhase.MIXED


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
        output = ScheduleOutput(
            running=running,
            preempted=preempted or [],
            scheduled=running if not preempted else [],
            connector_metadata=KVConnectorMetadata(),
            phase=step_phase(running),
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
    """Continuous batching with paged KV blocks and chunked prefill.

    Ported from vLLM's V1 scheduler: every step has a token budget
    (``max_num_batched_tokens``). Running requests are served first in arrival
    order, decodes taking one token each and in-progress prefills or recomputes
    taking the next chunk of their context; the remaining budget admits
    waiting requests, whose first chunk may also be partial. ``None`` disables
    the budget, so every prompt is prefilled in one step (the legacy
    behaviour).
    """

    def __init__(
        self,
        id: int,
        cache_config: CacheConfig,
        max_parallem_sum=None,
        max_occupy_ratio: float = 1,
        connector=None,
        max_num_batched_tokens: int | None = DEFAULT_MAX_NUM_BATCHED_TOKENS,
    ):
        super().__init__(connector=connector)

        self.id = id
        self.cache_config = cache_config
        self.max_parallem_sum = max_parallem_sum
        if max_num_batched_tokens is not None and max_num_batched_tokens < 1:
            raise ConfigurationError("max_num_batched_tokens must be at least 1")
        self.max_num_batched_tokens = max_num_batched_tokens
        # Requests admitted by the most recent schedule() call.
        self.last_admitted: list[Request] = []

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
        """Build one step.

        Returns the requests computing tokens in this step (each with
        ``scheduled_tokens`` set) and the requests preempted while making room
        for decode slots. Running requests that got no budget stay in
        ``self.running`` untouched; admissions of this step are recorded in
        ``self.last_admitted``.
        """
        budget = self.max_num_batched_tokens
        scheduled: list[Request] = []
        preempted: list[Request] = []
        self.last_admitted = []

        # 1. Running requests, oldest first: decode tokens and context chunks.
        # [TODO: xuechao] sort self.running according to some priority policy.
        queue = sorted(self.running, key=lambda req: req.arrival_time or 0.0)
        self.running = []
        while queue:
            req = queue.pop(0)
            tokens = self._step_tokens(req, budget)
            if tokens == 0:
                # Out of budget: the request keeps its place for the next step.
                self.running.append(req)
                continue
            if not req.is_context_build and not self._reserve_decode_slot(req, queue, preempted):
                continue
            req.scheduled_tokens = tokens
            budget = None if budget is None else budget - tokens
            scheduled.append(req)
            self.running.append(req)

        # 2. Admissions while budget and KV blocks last (head-of-line order).
        while self.waiting and (budget is None or budget > 0):
            if not self._is_occupy_below_usage():
                break
            if (
                self.max_parallem_sum is not None
                and len(self.running) >= self.max_parallem_sum
            ):
                break
            req = self.waiting[0]
            if not self.block_manager.can_allocate(req):
                break

            req = self.waiting.pop(0)
            self._admit(req)
            tokens = self._step_tokens(req, budget)
            req.scheduled_tokens = tokens
            budget = None if budget is None else budget - tokens
            scheduled.append(req)
            self.running.append(req)
            self.last_admitted.append(req)

        return scheduled, preempted

    def _admit(self, req: Request) -> None:
        """Allocate KV blocks for the whole context and record what is already cached."""
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
        if req.needs_recompute:
            to_compute = req.recompute_tokens
        elif req.is_prefill:
            to_compute = req.prefill_compute_len
        else:
            # KV arrived through a connector (prefill/decode disaggregation):
            # only the newest token is computed here.
            to_compute = 1
        req.start_context_build(req.context_len - to_compute)
        req.status = RequestStatus.RUNNING

    def _step_tokens(self, req: Request, budget: int | None) -> int:
        """Tokens ``req`` computes this step: its remaining context, capped by the budget."""
        remaining = max(1, req.remaining_context_tokens)
        return remaining if budget is None else min(remaining, budget)

    def _reserve_decode_slot(
        self,
        req: Request,
        queue: list[Request],
        preempted: list[Request],
    ) -> bool:
        """Find a KV slot for the token ``req`` computes this step.

        Preempts the youngest not-yet-scheduled running requests until a block
        is free; preempts ``req`` itself when nobody else is left. Returns
        whether ``req`` may run.
        """
        while True:
            if self.block_manager.can_append_slot(req):
                self.block_manager.append_slot(req)
                return True
            if queue:
                victim = queue.pop(-1)
                self._preempt(victim)
                preempted.append(victim)
            else:
                self._preempt(req)
                preempted.append(req)
                return False

    def schedule_output(self) -> ScheduleOutput:
        scheduled, preempted = self.schedule()
        output = ScheduleOutput(
            running=scheduled,
            preempted=preempted,
            scheduled=list(self.last_admitted),
            connector_metadata=KVConnectorMetadata(
                preempted_request_ids=[req.id for req in preempted],
            ),
            phase=step_phase(scheduled),
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
