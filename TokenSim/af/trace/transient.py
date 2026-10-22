"""AF validation adapter over shared trace transient."""

from TokenSim.trace.transient import ValidationTraceRecorder as _Recorder
from TokenSim.af.trace.recorder import AFTraceRecorder


class ValidationTraceRecorder(_Recorder, AFTraceRecorder):
    """Retain AF summaries and invariant checks with shared storage."""
