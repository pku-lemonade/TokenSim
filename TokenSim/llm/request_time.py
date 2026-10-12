"""Online request timing aggregates and run-wide timing views."""

from __future__ import annotations

from pydantic import BaseModel


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
