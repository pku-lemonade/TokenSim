"""AF validation adapter over shared trace msgpack."""

from TokenSim.trace.msgpack import MessagePackTraceRecorder as _Recorder
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.trace.msgpack import MessagePackWriter, HEADER, require_msgpack, convert


class MessagePackTraceRecorder(_Recorder, AFTraceRecorder):
    """Retain AF summaries and invariant checks with shared storage."""
