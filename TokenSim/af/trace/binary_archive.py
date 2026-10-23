"""AF validation adapter over shared trace binary archive."""

from TokenSim.trace.binary_archive import ArchiveTraceRecorder as _Recorder
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.trace.binary_archive import write_archive, convert, FORMAT, LEGACY_FORMAT


class ArchiveTraceRecorder(_Recorder, AFTraceRecorder):
    """Retain AF summaries and invariant checks with shared storage."""
