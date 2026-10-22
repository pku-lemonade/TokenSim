"""AF validation adapter over shared trace archive."""

from TokenSim.trace.archive import ArchivedTraceRecorder as _Recorder
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.trace.archive import ArchivedEventHistory, ArchivedIterationIndex


class ArchivedTraceRecorder(_Recorder, AFTraceRecorder):
    """Retain AF summaries and invariant checks with shared storage."""
