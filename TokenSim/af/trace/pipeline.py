"""AF validation adapter over shared trace pipeline."""

from TokenSim.trace.pipeline import PipelinedTraceRecorder as _Recorder
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.trace.pipeline import _PrefixWriter, _PreparedHistory


class PipelinedTraceRecorder(_Recorder, AFTraceRecorder):
    """Retain AF summaries and invariant checks with shared storage."""
