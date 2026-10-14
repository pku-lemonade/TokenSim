"""Worker messages, stable queue ordering and completion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from enum import Enum, auto
from typing import Any, Optional
import time
import simpy
from TokenSim.kv_transfer import KVConnectorMetadata
from TokenSim.llm.llm_request import Request


_message_sequence = count()


class Task(Enum):
    CANCEL = 0
    ADD = auto()
    STEP = auto()
    STOP = auto()


@dataclass
class Message:
    task: Task
    sender_id: int = 0
    requests: list[Request] | None = None
    metadata: Any | None = None
    sequence: int = field(
        default_factory=lambda: next(_message_sequence), init=False, repr=False
    )

    def __lt__(self, other: Message):
        return (self.task.value, self.sequence) < (other.task.value, other.sequence)


class WorkerStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    STOPPED = auto()


class RequestCompletionDebugPrinter:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.wall_start: float | None = None

    def start(self, wall_start: float | None = None) -> None:
        self.wall_start = time.perf_counter() if wall_start is None else wall_start

    def record(self, event: str, request: Request, timestamp: float) -> None:
        if not self.enabled or self.wall_start is None:
            return
        wall_elapsed = time.perf_counter() - self.wall_start
        print(
            f"[request={request.id}] debug event={event} "
            f"wall_elapsed_s={wall_elapsed:.6f} "
            f"request_id={request.id} input_len={request.prefill_len} "
            f"output_len={request.generation_idx} timestamp={timestamp:.6f}"
        )


class Worker:
    def __init__(self, env: simpy.Environment, id: int) -> None:
        self.env = env
        self.id = id
        self.msg_queue = simpy.PriorityStore(self.env)

    def send_task(
        self,
        worker: Worker,
        task: Task,
        requests: Optional[list[Request]] = None,
        metadata=None,
    ):
        if task == Task.ADD and worker.id >= 0:
            for request in requests or []:
                request.owner_worker_ids.add(worker.id)
        worker.msg_queue.put(Message(task, self.id, requests, metadata))


def _request_ids(requests: list[Request] | None) -> list[int]:
    if not requests:
        return []
    return [req.id for req in requests]


def _merge_connector_metadata(
    left: KVConnectorMetadata,
    right: KVConnectorMetadata | None,
) -> KVConnectorMetadata:
    if right is None:
        return left
    return KVConnectorMetadata(
        connector_name=right.connector_name if right.has_work else left.connector_name,
        loads=[*left.loads, *right.loads],
        saves=[*left.saves, *right.saves],
        preempted_request_ids=[
            *left.preempted_request_ids,
            *right.preempted_request_ids,
        ],
    )
