"""Trace representation selection and streaming conversion to Chrome JSON."""

from contextlib import closing
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from TokenSim.trace.export import _export_options, write_chrome_trace
from TokenSim.errors import ConfigurationError

TRACE_FORMATS = ("json", "msgpack", "archive")
TRACE_SUFFIXES = dict(zip(TRACE_FORMATS, (".json", ".msgpack", ".sqlite3")))


def validate_trace_format(value, *, require_dependency=False):
    if not isinstance(value, str) or value not in TRACE_FORMATS:
        raise ConfigurationError("trace_format must be json, msgpack or archive")
    if require_dependency and value == "msgpack":
        from TokenSim.trace.msgpack import require_msgpack

        require_msgpack()
    return value


def make_trace_recorder(directory, options, trace_format, *, recorder_classes=None):
    validate_trace_format(trace_format, require_dependency=True)
    if trace_format == "json":
        from TokenSim.trace.pipeline import PipelinedTraceRecorder

        kind = PipelinedTraceRecorder
    elif trace_format == "msgpack":
        from TokenSim.trace.msgpack import MessagePackTraceRecorder

        kind = MessagePackTraceRecorder
    else:
        from TokenSim.trace.binary_archive import ArchiveTraceRecorder

        kind = ArchiveTraceRecorder
    if recorder_classes is not None:
        kind = recorder_classes[trace_format]
    return kind(directory, trace_config=options)


def write_trace(path, events, metadata=None, *, trace_config=None, trace_format="json"):
    """Publish the selected format, including after prepared output is invalidated."""
    validate_trace_format(trace_format, require_dependency=True)
    if trace_format == "json":
        return write_chrome_trace(path, events, metadata, trace_config=trace_config)
    options, metadata = _export_options(trace_config, metadata)
    output = Path(path).with_suffix(TRACE_SUFFIXES[trace_format])
    if trace_format == "archive":
        from TokenSim.trace.binary_archive import write_archive

        return write_archive(output, events, metadata, options)
    from TokenSim.trace.msgpack import MessagePackWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    if getattr(events, "trace_format", None) == "msgpack":
        prepared = events._write_prepared_trace(output, metadata, options)
        if prepared is not None:
            return prepared
    with closing(MessagePackWriter(output.parent, options)) as writer:
        writer.append(events)
        return writer.finish(output, metadata)


def convert_trace_to_json(source, output, *, trace_format=None):
    """Convert a saved trace without simulation; preserve output on failure.

    The format is inferred from the saved filename unless supplied explicitly.
    JSON input is copied in bounded memory and may refer to the output itself.
    """
    if not isinstance(source, (str, Path)) or not isinstance(output, (str, Path)):
        raise ConfigurationError("trace source and output must be paths")
    source, output = Path(source), Path(output)
    if trace_format is None:
        trace_format = next(
            (
                name
                for name, suffix in TRACE_SUFFIXES.items()
                if source.suffix == suffix
            ),
            None,
        )
    validate_trace_format(trace_format, require_dependency=True)
    if trace_format != "json":
        if trace_format == "msgpack":
            from TokenSim.trace.msgpack import convert
        else:
            from TokenSim.trace.binary_archive import convert
        convert(source, output)
        return output
    if source.resolve() == output.resolve():
        if not source.is_file():
            raise FileNotFoundError(source)
        return output
    temporary = None
    try:
        with source.open("rb") as stream:
            output.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="wb", dir=output.parent, suffix=".tmp", delete=False
            ) as destination:
                temporary = Path(destination.name)
                shutil.copyfileobj(stream, destination, length=1024 * 1024)
        temporary.replace(output)
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
