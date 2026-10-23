"""Import simulated and measured device spans with provenance."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.records import Span, canonical
from TokenSim.profiler.io import _read, _complete, _manifest
from TokenSim.profiler.attribution import _observed


@dataclass
class ImportedTrace:
    spans: list[Span]
    metadata: dict
    inputs: list[dict]
    diagnostics: dict


def _simulated(path, rows, aliases):
    spans = []
    scopes = defaultdict(dict)
    seen = set()
    for row in rows:
        args = row.get("args", {})
        details = args.get("details", {})
        if any(
            key not in args
            for key in ("node_id", "worker_id", "layer_id", "iteration_id")
        ):
            raise ConfigurationError(
                "simulated comparison requires node_id, worker_id, layer_id and iteration_id event fields"
            )
        details = dict(
            details,
            run_phase=args.get("run_phase", details.get("run_phase", "profiling")),
            incomplete=args.get("incomplete", details.get("incomplete", False)),
        )
        operation = (
            args.get("operation") or args.get("stage") or row.get("name", "unknown")
        )
        if details.get("accounting_role") == "container":
            continue
        rank = str(args.get("worker_id"))
        execution = args.get("execution_group_id")
        if execution is not None:
            key = execution, rank
            if key in seen:
                continue
            seen.add(key)
        category = args.get("event_type", row.get("cat", "compute"))
        forward = None
        if (
            operation not in ("kv_transfer", "kv_save", "kv_dma", "af_completion")
            and details.get("run_phase") != "warmup"
        ):
            node = str(args.get("node_id", ""))
            scope = (
                ":".join(node.split(":")[:3])
                if node.startswith("forward:")
                else "af:" + str(args.get("iteration_id"))
            )
            if scope not in scopes[rank]:
                scopes[rank][scope] = len(scopes[rank])
            forward = scopes[rank][scope]
        spans.append(
            Span(
                id=args.get("node_id") or f"simulated:{len(spans)}",
                operation=canonical(operation, aliases),
                rank=rank,
                forward=forward,
                layer=args.get("layer_id"),
                phase=args.get("phase"),
                start_us=row["ts"],
                end_us=row["ts"] + row["dur"],
                category=category,
                source=str(path),
                clock="simulation",
                kernel_sum_us=row["dur"],
                busy_us=row["dur"],
                shape=details.get("shape", {}),
                details=details,
            )
        )
        if args.get("token_count", 0) > 0:
            spans[-1].shape.setdefault("forward_tokens", args["token_count"])
    return spans


def read_trace(paths, *, simulated=False, aliases=None, manifest_path=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    spans, inputs, metadata, diagnostics = [], [], {}, {}
    for index, path in enumerate(paths):
        raw, trace_metadata, digest = _read(path)
        rows, ignored = _complete(raw)
        manifest = (
            {}
            if simulated
            else (
                _manifest(manifest_path, exact=True)
                if manifest_path is not None
                else _manifest(path)
            )
        )
        if simulated:
            current = _simulated(path, rows, aliases)
        else:
            from TokenSim.profiler.paired import paired_observed

            current = paired_observed(
                path,
                rows,
                digest,
                manifest,
                aliases,
                index,
                root=(
                    Path(manifest_path).resolve().parent
                    if manifest_path is not None
                    else None
                ),
            )
            if current is None:
                if (
                    manifest_path is not None
                    and manifest.get("measurement_protocol")
                    == "paired_structure_gpu_timing_v1"
                ):
                    raise ConfigurationError(
                        "selected timing capture lacks a verified structure binding"
                    )
                current = _observed(path, rows, aliases, index, manifest)
        spans.extend(current)
        inputs.append(
            dict(
                path=str(path),
                sha256=digest,
                manifest=manifest,
                metadata=trace_metadata,
            )
        )
        metadata.update(trace_metadata)
        diagnostics[str(path)] = dict(ignored_events=ignored, device_spans=len(current))
    return ImportedTrace(spans, metadata, inputs, diagnostics)
