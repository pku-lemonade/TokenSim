"""Compatibility access to shared trace formats."""

from TokenSim.trace.formats import (
    TRACE_FORMATS,
    TRACE_SUFFIXES,
    validate_trace_format,
    write_trace,
    convert_trace_to_json,
)

from TokenSim.trace import formats as _shared


def make_trace_recorder(directory, options, trace_format):
    from TokenSim.af.trace.pipeline import PipelinedTraceRecorder
    from TokenSim.af.trace.msgpack import MessagePackTraceRecorder
    from TokenSim.af.trace.binary_archive import ArchiveTraceRecorder

    return _shared.make_trace_recorder(
        directory,
        options,
        trace_format,
        recorder_classes={
            "json": PipelinedTraceRecorder,
            "msgpack": MessagePackTraceRecorder,
            "archive": ArchiveTraceRecorder,
        },
    )
