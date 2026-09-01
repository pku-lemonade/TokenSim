import simpy
from dataclasses import dataclass, field
from pydantic import BaseModel
from enum import Enum, auto

from TokenSim.block.block import LogicalTokenBlock, PhysicalTokenBlock


class RequestStatus(Enum):
    """Status of a sequence."""

    WAITING = auto()
    RUNNING = auto()
    WAITING_FOR_KV = auto()
    WAITING_FOR_CONNECTOR_FREE = auto()
    FINISHED_STOPPED = auto()


class RequestTime(BaseModel):
    """Per-request timing aggregates.

    Historically this stored the raw per-step ``time``/``service_time``/
    ``batch`` lists; they grew with decode length and dominated result memory
    at trace scale, so the same statistics are aggregated online instead.
    Aggregation order matches the old list arithmetic bit-for-bit.
    """

    id: int
    steps: int
    total_time: float = 0.0
    first_step_time: float = 0.0
    decode_time_sum: float = 0.0
    decode_time_max_value: float = 0.0
    first_step_service_time: float = 0.0
    decode_service_time_sum: float = 0.0
    first_step_batch: float = 0.0
    decode_batch_sum: float = 0.0

    @classmethod
    def from_step_lists(
        cls,
        id: int,
        time: list[float],
        service_time: list[float],
        batch: list[int | float],
    ) -> "RequestTime":
        """Build aggregates from per-step delta lists (test/back-compat helper)."""
        decode_sum = 0.0
        for value in time[1:]:
            decode_sum += value
        decode_service_sum = 0.0
        for value in service_time[1:]:
            decode_service_sum += value
        decode_batch_sum = 0.0
        for value in batch[1:]:
            decode_batch_sum += value
        total = 0.0
        for value in time:
            total += value
        return cls(
            id=id,
            steps=len(time),
            total_time=total,
            first_step_time=time[0] if time else 0.0,
            decode_time_sum=decode_sum,
            decode_time_max_value=max(time[1:]) if len(time) > 1 else 0.0,
            first_step_service_time=service_time[0] if service_time else 0.0,
            decode_service_time_sum=decode_service_sum,
            first_step_batch=batch[0] if batch else 0.0,
            decode_batch_sum=decode_batch_sum,
        )

    @property
    def request_time(self):
        return self.total_time

    @property
    def prompt_time(self):
        return self.first_step_time

    @property
    def prefill_time(self):
        return self.first_step_time

    @property
    def decode_time(self):
        # Requests with a single output token have no inter-token gaps.
        if self.steps <= 1:
            return 0.0
        return self.decode_time_sum / (self.steps - 1)

    @property
    def decode_max_time(self):
        if self.steps <= 1:
            return 0.0
        return self.decode_time_max_value

    @property
    def prompt_batch(self):
        return self.first_step_batch

    @property
    def decode_batch(self):
        if self.steps <= 1:
            return 0.0
        return self.decode_batch_sum / (self.steps - 1)

    @property
    def prompt_util(self):
        return self.first_step_service_time / self.first_step_time

    @property
    def prefill_util(self):
        return self.prompt_util

    @property
    def decode_util(self):
        if self.decode_time_sum <= 0:
            return 0.0
        return self.decode_service_time_sum / self.decode_time_sum

    @property
    def prompt_idle(self):
        return 1 - self.prompt_util

    @property
    def prefill_idle(self):
        return 1 - self.prefill_util

    @property
    def decode_idle(self):
        return 1 - self.decode_util


class LLMTime(BaseModel):
    time: list[RequestTime] = []

    @property
    def request_time(self):
        return [t.request_time for t in self.time]

    @property
    def prompt_time(self):
        return [t.prompt_time for t in self.time]

    @property
    def prefill_time(self):
        return [t.prefill_time for t in self.time]

    @property
    def decode_time(self):
        return [t.decode_time for t in self.time]

    @property
    def decode_max_time(self):
        return [t.decode_max_time for t in self.time]

    @property
    def prompt_util(self):
        return [t.prompt_util for t in self.time]

    @property
    def prefill_util(self):
        return [t.prefill_util for t in self.time]

    @property
    def decode_util(self):
        return [t.decode_util for t in self.time]

    @property
    def prompt_idle(self):
        return [t.prompt_idle for t in self.time]

    @property
    def prefill_idle(self):
        return [t.prefill_idle for t in self.time]

    @property
    def decode_idle(self):
        return [t.decode_idle for t in self.time]

    @property
    def prompt_batch(self):
        return [t.prompt_batch for t in self.time]

    @property
    def decode_batch(self):
        return [t.decode_batch for t in self.time]


g_time = LLMTime()


@dataclass
class Request:
    id: int
    prefill_len: int
    decode_len: int
    block_size: int = field(repr=False)
    arrival_timestamp: float | None = None
    inter_arrival_time: float | None = None
    chat_id: str | None = None
    parent_chat_id: str | None = None
    turn: int | None = None
    hash_ids: list[str | int] | None = None
    output_hash_ids: list[str | int] | None = None
    expert_histogram: dict[int, int] | list | None = None
    prefill_expert_histogram: dict[int, int] | list | None = None
    decode_expert_histogram: dict[int, int] | list | None = None
    cache_salt: str | None = None
    reuse_group: str | None = None
    cached_prefill_blocks: int = 0
    cached_prefill_tokens: int = 0
    effective_prefill_tokens: int | None = None
    reuse_hit_blocks: int = 0
    reuse_miss_blocks: int = 0
    input_cache_committed: bool = False
    input_cache_keys: list = field(default_factory=list, repr=False)
    generation_idx: int = 0
    needs_recompute: bool = field(default=False, repr=False)
    recompute_tokens: int = 0
    recomputation_count: int = 0
    recomputed_tokens_total: int = 0
    recompute_service_time: float = 0.0
    status: RequestStatus = field(default=RequestStatus.WAITING, repr=False)
    # Online timing aggregates (replacing per-step time/service_time/batch
    # lists; accumulation order matches the old list arithmetic bit-for-bit).
    arrival_at: float | None = field(default=None, repr=False)
    total_time: float = field(default=0.0, repr=False)
    prefill_latency: float = field(default=0.0, repr=False)
    decode_time_sum: float = field(default=0.0, repr=False)
    decode_time_max: float = field(default=0.0, repr=False)
    prefill_service_time: float = field(default=0.0, repr=False)
    decode_service_time_sum: float = field(default=0.0, repr=False)
    prefill_batch_size: float = field(default=0.0, repr=False)
    decode_batch_sum: float = field(default=0.0, repr=False)
    tqdm_submit_func = None

    def __post_init__(self):
        self._physical_token_blocks: list[PhysicalTokenBlock] = []
        self._logical_token_blocks: list[LogicalTokenBlock] = []
        self._append_tokens(self.prefill_len)
        self._last_event_at: float = 0.0
        if self.effective_prefill_tokens is None:
            self.effective_prefill_tokens = self.prefill_len

    @property
    def prompt_len(self) -> int:
        return self.prefill_len

    @property
    def generation_len(self) -> int:
        return self.decode_len

    @property
    def prefill_compute_len(self) -> int:
        if self.effective_prefill_tokens is None:
            return self.prefill_len
        return self.effective_prefill_tokens

    @property
    def is_prefill(self) -> bool:
        return self.generation_idx == 0

    @property
    def is_prompt(self) -> bool:
        return self.is_prefill

    @property
    def is_decode(self) -> bool:
        return not self.is_prefill

    @property
    def context_len(self) -> int:
        return self.prefill_len + self.generation_idx

    @property
    def is_done(self) -> bool:
        return self.generation_idx == self.decode_len

    @property
    def num_physical_token_blocks(self):
        return len(self._physical_token_blocks)

    @property
    def num_logical_token_blocks(self):
        return len(self._logical_token_blocks)

    def _append_physical_block(self, block) -> None:
        self._physical_token_blocks.append(block)

    def _append_logical_block(self) -> None:
        """Create a new logical block and append it at the end of all blocks."""
        block = LogicalTokenBlock(
            block_id=self.num_logical_token_blocks,
            block_size=self.block_size,
        )
        self._logical_token_blocks.append(block)

    def _append_tokens(self, num_tokens: int) -> None:
        cursor = 0
        while cursor < num_tokens:
            if not self._logical_token_blocks:
                self._append_logical_block()

            last_block = self._logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self._logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(min(num_empty_slots, num_tokens - cursor))
            cursor += num_empty_slots

    @property
    def arrival_time(self):
        return self.arrival_at

    def arrive(self, env: simpy.Environment):
        self.arrival_at = env.now
        self._last_event_at = env.now

    def step(
        self, env: simpy.Environment, latency: float, batch: int, timing_recorder=None
    ):
        self.generation_idx += 1
        self._append_tokens(1)
        step_time = env.now - self._last_event_at
        self._last_event_at = env.now
        self.total_time += step_time
        if self.generation_idx == 1:
            self.prefill_latency = step_time
            self.prefill_service_time = latency
            self.prefill_batch_size = batch
        else:
            self.decode_time_sum += step_time
            if step_time > self.decode_time_max:
                self.decode_time_max = step_time
            self.decode_service_time_sum += latency
            self.decode_batch_sum += batch
        if self.is_done:
            request_time = self.complete_timing()
            if timing_recorder is None:
                from TokenSim.timing import DEFAULT_TIMING_RECORDER

                timing_recorder = DEFAULT_TIMING_RECORDER
            timing_recorder.record_request(request_time)
            if self.tqdm_submit_func:
                self.tqdm_submit_func(1)
            return request_time
        return None

    def prepare_recompute(self) -> None:
        self.needs_recompute = True
        self.recompute_tokens = self.context_len

    def finish_recompute(self, latency: float) -> None:
        self.needs_recompute = False
        self.recomputation_count += 1
        self.recomputed_tokens_total += self.recompute_tokens
        self.recompute_service_time += latency
        self.recompute_tokens = 0

    def complete_timing(self) -> RequestTime:
        return RequestTime(
            id=self.id,
            steps=self.generation_idx,
            total_time=self.total_time,
            first_step_time=self.prefill_latency,
            decode_time_sum=self.decode_time_sum,
            decode_time_max_value=self.decode_time_max,
            first_step_service_time=self.prefill_service_time,
            decode_service_time_sum=self.decode_service_time_sum,
            first_step_batch=self.prefill_batch_size,
            decode_batch_sum=self.decode_batch_sum,
        )


def reset_g_time() -> LLMTime:
    from TokenSim.timing import reset_timing

    return reset_timing()
