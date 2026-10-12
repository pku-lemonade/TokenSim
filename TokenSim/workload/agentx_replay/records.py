"""Trace request identity and mutable replay tree state."""

from __future__ import annotations

from array import array
import hashlib
import sys
from dataclasses import dataclass, field
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.workload.agentx_support.models import Conversation, TraceGraph, TurnKey


REPLAY_PROTOCOL = "aiperf-49634d1c-native"


def engine_hashes(
    hashes, *, trace_block_size: int, engine_block_size: int, input_tokens: int
) -> list[str | int]:
    if engine_block_size % trace_block_size:
        raise WorkloadValidationError(
            "engine block size must be a multiple of trace hash block size"
        )
    width = engine_block_size // trace_block_size
    complete = min(len(hashes) // width, input_tokens // engine_block_size)
    if width == 1:
        return list(hashes[:complete])
    # Each engine block hashes its trace hashes as big-endian u64 values.
    words = array("Q", hashes[: complete * width])
    if sys.byteorder == "little":
        words.byteswap()
    data = words.tobytes()
    stride = 8 * width
    sha256 = hashlib.sha256
    return [
        sha256(data[i : i + stride]).hexdigest()
        for i in range(0, complete * stride, stride)
    ]


@dataclass(frozen=True, slots=True)
class TraceTurn:
    """Where an AgentX request came from in the trace pool."""

    trace_id: str
    play_id: int
    lane: int
    stream_id: str
    conversation_id: str
    outer_index: int
    inner_index: int | None
    recorded_model: str


@dataclass
class StreamCursor:
    conversation: Conversation
    index: int
    active: bool
    due: float | None
    in_flight: Request | None = None
    terminal: bool = False
    ordinal: int = 0


@dataclass
class TreePlay:
    graph: TraceGraph
    lane: int
    pass_index: int
    origin_s: float
    streams: dict[str, StreamCursor]
    completed: set[TurnKey]
    opened_branches: set[str]
    scope: str
    in_flight: int = 0
    clock_origin_s: float = 0.0
    idle_started_at: float | None = None
    idle_cap_expired: bool = False
    active_streams: dict[str, StreamCursor] = field(default_factory=dict)
    unfinished_streams: int = 0

    def __post_init__(self):
        self.active_streams = {
            cid: c for cid, c in self.streams.items() if c.active and not c.terminal
        }
        self.unfinished_streams = sum(not c.terminal for c in self.streams.values())

    def set_terminal(self, cursor: StreamCursor) -> None:
        if not cursor.terminal:
            cursor.terminal = True
            self.unfinished_streams -= 1
        self.active_streams.pop(cursor.conversation.conversation_id, None)

    def finished(self) -> bool:
        return self.in_flight == 0 and self.unfinished_streams == 0
