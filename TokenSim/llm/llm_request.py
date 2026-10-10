import simpy
from dataclasses import dataclass, field
from typing import Sequence
from pydantic import BaseModel
from enum import Enum, auto

from TokenSim.block.block import LogicalTokenBlock, PhysicalTokenBlock
from TokenSim.errors import SimulationStateError


class RequestStatus(Enum):
    """Status of a sequence."""

    WAITING = auto()
    RUNNING = auto()
    WAITING_FOR_KV = auto()
    WAITING_FOR_CONNECTOR_FREE = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    CANCEL_PENDING = auto()
    FINISHED_CANCELLED = auto()


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


@dataclass(frozen=True)
class PrefillSourceStats:
    """Original prompt sources, independent of post-preemption KV recovery."""

    input_tokens: int
    gpu_hit_tokens: int
    cpu_hit_tokens: int

    @property
    def computed_tokens(self):
        return self.input_tokens - self.gpu_hit_tokens - self.cpu_hit_tokens


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
    measurement_phase: str = "profile"
    record_timing: bool = True
    cached_prefill_blocks: int = 0
    cached_prefill_tokens: int = 0
    gpu_cached_prefill_tokens: int = 0
    external_cached_prefill_tokens: int = 0
    gpu_cached_context_tokens: int = 0
    external_cached_context_tokens: int = 0
    first_prefill_source: PrefillSourceStats | None = field(default=None, repr=False)
    trace_hash_ids: Sequence[int] | None = field(default=None, repr=False)
    trace_block_size: int | None = field(default=None, repr=False)
    shared_prefix_boundaries: tuple[int, ...] = ()
    prefill_admission_count: int = 0
    reloaded_gpu_tokens: int = 0
    reloaded_cpu_tokens: int = 0
    reloaded_generated_gpu_tokens: int = 0
    reloaded_generated_cpu_tokens: int = 0
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
    completed_at: float | None = field(default=None, repr=False)
    total_time: float = field(default=0.0, repr=False)
    prefill_latency: float = field(default=0.0, repr=False)
    decode_time_sum: float = field(default=0.0, repr=False)
    decode_time_max: float = field(default=0.0, repr=False)
    prefill_service_time: float = field(default=0.0, repr=False)
    decode_service_time_sum: float = field(default=0.0, repr=False)
    prefill_batch_size: float = field(default=0.0, repr=False)
    decode_batch_sum: float = field(default=0.0, repr=False)
    # Per-step scheduling state (vLLM V1 semantics). ``num_computed_tokens`` is
    # the number of context tokens whose KV is present on the worker (prefix
    # hits included); ``scheduled_tokens`` is what the scheduler assigned to
    # the current step: one token for a decode step, the next chunk of the
    # remaining context for a (chunked) prefill or recompute step.
    num_computed_tokens: int = field(default=0, repr=False)
    scheduled_tokens: int = field(default=0, repr=False)
    # MTP advances by the accepted 2/3 tokens while the target forward
    # verifies 1+K query positions.  Ordinary requests keep this equal to
    # scheduled_tokens.
    scheduled_query_tokens: int = field(default=0, repr=False)
    scheduled_placeholder_tokens: int = field(default=0, repr=False)
    prefill_chunk_pending: bool = field(default=False, repr=False)
    max_model_len: int | None = field(default=None, repr=False)
    cancel_requested: bool = field(default=False, repr=False)
    cancel_requested_at: float | None = field(default=None, repr=False)
    cancelled_at: float | None = field(default=None, repr=False)
    owner_worker_ids: set[int] = field(default_factory=set, repr=False)
    cancel_pending_worker_ids: set[int] = field(default_factory=set, repr=False)
    # Context-building steps this request's prompt took (1 without chunking).
    prefill_chunks: int = 0
    tqdm_submit_func = None

    def __post_init__(self):
        self._physical_token_blocks: list[PhysicalTokenBlock] = []
        self._logical_token_blocks: list[LogicalTokenBlock] = []
        self._append_tokens(self.prefill_len)
        self._last_event_at: float = 0.0
        self.completion_event: simpy.Event | None = None
        self.cancellation_event: simpy.Event | None = None
        if self.effective_prefill_tokens is None:
            self.effective_prefill_tokens = self.prefill_len

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
        return not self.cancel_requested and (
            self.generation_idx >= self.decode_len
            or (
                self.max_model_len is not None
                and self.context_len >= self.max_model_len
            )
        )

    def request_cancel(self, env: simpy.Environment) -> simpy.Event:
        if self.cancellation_event is None:
            self.cancellation_event = env.event()
        if self.is_done:
            if not self.cancellation_event.triggered:
                self.cancellation_event.succeed(self)
            return self.cancellation_event
        if not self.cancel_requested:
            self.cancel_requested = True
            self.cancel_requested_at = env.now
            self.status = RequestStatus.CANCEL_PENDING
            self.cancel_pending_worker_ids = set(self.owner_worker_ids)
        return self.cancellation_event

    def acknowledge_cancel(
        self, env: simpy.Environment, worker_id: int | None = None
    ) -> None:
        if not self.cancel_requested:
            return
        if worker_id is not None:
            self.cancel_pending_worker_ids.discard(worker_id)
        if self.cancel_pending_worker_ids or self.cancelled_at is not None:
            return
        self.cancelled_at = env.now
        self.status = RequestStatus.FINISHED_CANCELLED
        if self.completion_event is not None and not self.completion_event.triggered:
            self.completion_event.succeed(None)
        if (
            self.cancellation_event is not None
            and not self.cancellation_event.triggered
        ):
            self.cancellation_event.succeed(self)

    @property
    def remaining_context_tokens(self) -> int:
        """Context tokens still to compute before this request can sample a token."""
        return max(0, self.context_len - self.num_computed_tokens)

    @property
    def is_context_build(self) -> bool:
        """True while the prompt, or a preempted context, is being (re)computed."""
        return self.is_prefill or self.needs_recompute

    @property
    def is_prompt_prefilling(self) -> bool:
        """Serving's original-prompt progress, independent of output replay."""
        return self.is_context_build and self.num_computed_tokens < self.prefill_len

    @property
    def decode_graph_eligible(self) -> bool:
        return not self.is_prompt_prefilling or (
            self.step_tokens == 1 and self.num_computed_tokens > 0
        )

    @property
    def logits_tokens(self) -> int:
        # Invalid placeholder drafts still require logits but yield one output.
        return (
            1 + self.scheduled_placeholder_tokens
            if self.is_context_build
            else self.query_tokens
        )

    @property
    def step_tokens(self) -> int:
        """Tokens this request computes in the current step.

        The scheduler assigns ``scheduled_tokens`` (a decode token or a context
        chunk); a request nobody scheduled, as with the legacy schedulers or
        direct calls, computes its whole remaining context at once.
        """
        return self.scheduled_tokens or max(1, self.remaining_context_tokens)

    @property
    def query_tokens(self) -> int:
        """Target-model query width for the current scheduler step."""
        return self.scheduled_query_tokens or self.step_tokens

    def start_context_build(self, cached_tokens: int) -> None:
        """Record the context already present in KV when the request is admitted.

        At least one token is always computed (the last position's logits are
        needed), which is vLLM's cap on full-prompt prefix hits.
        """
        self.num_computed_tokens = min(
            max(0, cached_tokens), max(0, self.context_len - 1)
        )
        self.scheduled_tokens = 0
        self.scheduled_query_tokens = 0
        self.scheduled_placeholder_tokens = 0

    def record_prefill_admission(self, gpu_tokens, cpu_tokens):
        if (
            min(gpu_tokens, cpu_tokens) < 0
            or gpu_tokens + cpu_tokens > self.prefill_len
        ):
            raise SimulationStateError("prompt-source tokens exceed the original input")
        if self.first_prefill_source is None:
            self.first_prefill_source = PrefillSourceStats(
                self.prefill_len, gpu_tokens, cpu_tokens
            )
        else:
            self.reloaded_gpu_tokens += gpu_tokens
            self.reloaded_cpu_tokens += cpu_tokens
        self.prefill_admission_count += 1

    def record_context_admission(self, gpu_tokens, cpu_tokens):
        """Record a whole-context hit without expanding the input denominator."""
        if (
            min(gpu_tokens, cpu_tokens) < 0
            or gpu_tokens + cpu_tokens >= self.context_len
        ):
            raise SimulationStateError("context-source tokens must leave a real query")
        self.gpu_cached_context_tokens = gpu_tokens
        self.external_cached_context_tokens = cpu_tokens
        prompt_gpu = min(gpu_tokens, self.prefill_len)
        prompt_cpu = min(cpu_tokens, max(0, self.prefill_len - prompt_gpu))
        self.gpu_cached_prefill_tokens = prompt_gpu
        self.external_cached_prefill_tokens = prompt_cpu
        self.cached_prefill_tokens = prompt_gpu + prompt_cpu
        self.cached_prefill_blocks = self.cached_prefill_tokens // self.block_size
        self.effective_prefill_tokens = self.prefill_len - self.cached_prefill_tokens
        self.reloaded_generated_gpu_tokens += gpu_tokens - prompt_gpu
        self.reloaded_generated_cpu_tokens += cpu_tokens - prompt_cpu
        self.record_prefill_admission(prompt_gpu, prompt_cpu)

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
        if self.cancel_requested:
            return
        self.arrival_at = env.now
        self._last_event_at = env.now
        event_factory = getattr(env, "event", None)
        self.completion_event = event_factory() if event_factory is not None else None

    def advance(
        self, env: simpy.Environment, latency: float, batch: int, timing_recorder=None
    ):
        """Apply one scheduler step to this request.

        A step computes ``scheduled_tokens`` of context: a decode token, a
        prefill chunk or a recompute chunk. Only the step that completes the
        context samples a token (:meth:`step`); partial chunks just accumulate
        service time, so TTFT spans every chunk plus the waits between them.
        Returns the completed :class:`RequestTime` when the request finishes.
        """
        if self.cancel_requested:
            return None
        context_build = self.is_prefill or self.needs_recompute
        tokens = self.step_tokens
        self.scheduled_tokens = 0
        self.scheduled_query_tokens = 0
        self.scheduled_placeholder_tokens = 0
        self.num_computed_tokens += tokens
        if self.is_prefill:
            self.prefill_chunks += 1
        if self.num_computed_tokens < self.context_len:
            if self.needs_recompute:
                self.recompute_service_time += latency
            else:
                self.prefill_service_time += latency
            return None
        if self.needs_recompute:
            # The pass that restores the context also yields the next token;
            # its service time is booked as recompute, not decode.
            self.finish_recompute(latency)
            latency = 0.0
        return self.step(
            env,
            latency,
            batch,
            timing_recorder,
            output_tokens=(1 if context_build else tokens),
        )

    def step(
        self,
        env: simpy.Environment,
        latency: float,
        batch: int,
        timing_recorder=None,
        *,
        output_tokens: int = 1,
    ):
        """Sample one token: bookkeeping for wall time, service time and batch size."""
        if self.cancel_requested:
            return None
        output_tokens = max(
            1, min(output_tokens, self.decode_len - self.generation_idx)
        )
        if self.max_model_len is not None:
            output_tokens = min(
                output_tokens, max(0, self.max_model_len - self.context_len)
            )
        self.generation_idx += output_tokens
        self._append_tokens(output_tokens)
        step_time = env.now - self._last_event_at
        self._last_event_at = env.now
        self.total_time += step_time
        if self.generation_idx == 1:
            self.prefill_latency = step_time
            self.prefill_service_time += latency
            self.prefill_batch_size = batch
        else:
            self.decode_time_sum += step_time
            if step_time > self.decode_time_max:
                self.decode_time_max = step_time
            self.decode_service_time_sum += latency
            self.decode_batch_sum += batch
        if self.is_done:
            if (
                self.max_model_len is not None
                and self.context_len >= self.max_model_len
            ):
                self.status = RequestStatus.FINISHED_LENGTH_CAPPED
            self.completed_at = env.now
            request_time = self.complete_timing()
            if self.record_timing and timing_recorder is not None:
                timing_recorder.record_request(request_time)
            if (
                self.completion_event is not None
                and not self.completion_event.triggered
            ):
                self.completion_event.succeed(request_time)
            if self.tqdm_submit_func:
                self.tqdm_submit_func(1)
            return request_time
        return None

    def release_logical_blocks(self) -> None:
        """Drop bulky request-local state after the scheduler has finished it."""
        self._logical_token_blocks.clear()
        self.input_cache_keys.clear()

    def prepare_recompute(self) -> None:
        """Preemption released the KV: the whole context must be rebuilt."""
        self.needs_recompute = True
        self.recompute_tokens = self.context_len
        self.num_computed_tokens = 0
        self.scheduled_tokens = 0
        self.scheduled_query_tokens = 0
        self.scheduled_placeholder_tokens = 0

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
