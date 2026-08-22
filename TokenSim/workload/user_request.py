from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request


@dataclass
class UserRequest:
    request_id: int
    prefill_len: int
    decode_len: int
    arrival_time: float | None = None
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

    def __post_init__(self):
        self.request_id = int(self.request_id)
        self.prefill_len = _positive_int(self.prefill_len, "prefill_len")
        self.decode_len = _positive_int(self.decode_len, "decode_len")
        if self.arrival_time is not None and self.inter_arrival_time is not None:
            raise WorkloadValidationError(
                "arrival_time and inter_arrival_time are mutually exclusive"
            )
        if self.arrival_time is not None:
            self.arrival_time = float(self.arrival_time)
        if self.inter_arrival_time is not None:
            self.inter_arrival_time = float(self.inter_arrival_time)
        if self.turn is not None:
            self.turn = int(self.turn)
        if self.hash_ids is not None:
            if not isinstance(self.hash_ids, list):
                raise WorkloadValidationError("hash_ids must be a list when provided")
        if self.output_hash_ids is not None:
            if not isinstance(self.output_hash_ids, list):
                raise WorkloadValidationError(
                    "output_hash_ids must be a list when provided"
                )

    def to_request(self, block_size: int, tqdm_submit_func=None) -> Request:
        request = Request(
            id=self.request_id,
            prefill_len=self.prefill_len,
            decode_len=self.decode_len,
            block_size=block_size,
            arrival_timestamp=self.arrival_time,
            inter_arrival_time=self.inter_arrival_time,
            chat_id=self.chat_id,
            parent_chat_id=self.parent_chat_id,
            turn=self.turn,
            hash_ids=self.hash_ids,
            output_hash_ids=self.output_hash_ids,
            expert_histogram=self.expert_histogram,
            prefill_expert_histogram=self.prefill_expert_histogram,
            decode_expert_histogram=self.decode_expert_histogram,
            cache_salt=self.cache_salt,
            reuse_group=self.reuse_group,
        )
        request.tqdm_submit_func = tqdm_submit_func
        return request


class UserRequestList(list[UserRequest]):
    def to_requests(self, block_size: int, tqdm_submit_func=None) -> list[Request]:
        return [
            user_request.to_request(
                block_size=block_size,
                tqdm_submit_func=tqdm_submit_func,
            )
            for user_request in self
        ]

    def prefill_lens(self) -> list[int]:
        return [user_request.prefill_len for user_request in self]

    def decode_lens(self) -> list[int]:
        return [user_request.decode_len for user_request in self]


def _positive_int(value: Any, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise WorkloadValidationError(f"{field_name} must be positive, got {value}")
    return value
