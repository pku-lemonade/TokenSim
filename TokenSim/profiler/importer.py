"""Read Chrome events and associate GPU work through observed CPU ownership."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.annotations import kernel_operation, fused_members
from TokenSim.profiler.records import Span, annotation, canonical, union_duration


@dataclass
class ImportedTrace:
    spans: list[Span]
    metadata: dict
    inputs: list[dict]
    diagnostics: dict


def _read(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, ValueError) as error:
        raise ConfigurationError(
            f"cannot read profiler trace {path}: {error}"
        ) from error
    rows = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ConfigurationError("trace must contain an event array")
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    if not isinstance(metadata, dict):
        raise ConfigurationError("trace metadata must be an object")
    if isinstance(document, dict) and "baseTimeNanoseconds" in document:
        metadata = dict(metadata, baseTimeNanoseconds=document["baseTimeNanoseconds"])
    return rows, metadata, digest.hexdigest()


def _complete(rows):
    stacks = defaultdict(list)
    result = []
    ignored = defaultdict(int)
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ConfigurationError("trace events must be objects")
        row = dict(raw, _index=index)
        phase = row.get("ph")
        thread = str(row.get("pid")), str(row.get("tid"))
        if phase == "B":
            if type(row.get("ts")) not in (int, float) or not math.isfinite(row["ts"]):
                raise ConfigurationError("trace begin timestamp must be finite")
            stacks[thread].append(row)
            continue
        if phase == "E":
            if not stacks[thread]:
                raise ConfigurationError("trace end event has no begin event")
            if type(row.get("ts")) not in (int, float) or not math.isfinite(row["ts"]):
                raise ConfigurationError("trace end timestamp must be finite")
            start = stacks[thread].pop()
            row = dict(start, ph="X", dur=row["ts"] - start["ts"])
        elif phase != "X":
            ignored[str(phase)] += 1
            continue
        if (
            any(
                type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                for key in ("ts", "dur")
            )
            or row["dur"] < 0
        ):
            raise ConfigurationError(
                "trace timestamps and durations must be finite and nonnegative"
            )
        if not isinstance(row.get("args", {}), dict):
            raise ConfigurationError("trace event args must be an object")
        result.append(row)
    if any(stacks.values()):
        raise ConfigurationError("trace contains unclosed begin events")
    return result, dict(ignored)


def _manifest(path):
    for folder in (Path(path).parent, Path(path).parent.parent):
        candidate = folder / "manifest.json"
        if candidate.is_file():
            try:
                manifest = json.loads(candidate.read_text())
                if not isinstance(manifest, dict) or not isinstance(
                    manifest.get("model_config", {}), dict
                ):
                    raise ConfigurationError(
                        "collection manifest and model_config must be objects"
                    )
                for key in (
                    "batch_size",
                    "input_tokens",
                    "output_tokens",
                    "tensor_parallel_size",
                ):
                    if key in manifest and (
                        type(manifest[key]) is not int or manifest[key] < 1
                    ):
                        raise ConfigurationError(
                            f"collection {key} must be a positive integer"
                        )
                return manifest
            except (OSError, ValueError) as error:
                raise ConfigurationError(
                    f"invalid collection manifest {candidate}: {error}"
                ) from error
    return {}


def _gpu(row):
    category = str(row.get("cat", "")).lower()
    return category in ("kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset")


def _rank(path, rows):
    match = re.search(r"_tp(\d+)", Path(path).name)
    if match:
        dp, pp = re.search(r"_dp(\d+)", Path(path).name), re.search(
            r"_pp(\d+)", Path(path).name
        )
        if (dp and dp[1] != "0") or (pp and pp[1] != "0"):
            return f"dp{dp[1] if dp else 0}:pp{pp[1] if pp else 0}:tp{match[1]}"
        return match[1]
    devices = {str(row.get("args", {}).get("device")) for row in rows if _gpu(row)} - {
        "None"
    }
    return next(iter(devices)) if len(devices) == 1 else "unknown"


def _observed(path, rows, aliases, file_index, manifest):
    cpu = [
        row
        for row in rows
        if not _gpu(row) and not str(row.get("cat", "")).startswith("gpu_")
    ]
    annotations = [row for row in cpu if annotation(str(row.get("name", "")))]
    roots = sorted(row["ts"] for row in cpu if row.get("name") == "tokensim.forward")
    by_thread = defaultdict(list)
    for row in annotations:
        by_thread[str(row.get("pid")), str(row.get("tid"))].append(row)
    ext = defaultdict(list)
    corr = defaultdict(list)
    for row in cpu:
        args = row.get("args", {})
        if "External id" in args:
            ext[str(args["External id"])].append(row)
        if "correlation" in args:
            corr[str(args["correlation"])].append(row)
    rank = _rank(path, rows)
    groups = defaultdict(list)
    unknown = []
    for row in rows:
        if not _gpu(row):
            continue
        args = row.get("args", {})
        candidates = ext.get(str(args.get("External id")), [])
        if not candidates:
            candidates = corr.get(str(args.get("correlation")), [])
            linked = [
                item
                for candidate in candidates
                for item in ext.get(
                    str(candidate.get("args", {}).get("External id")), []
                )
            ]
            candidates = linked or candidates
        semantic = [
            candidate for candidate in candidates if candidate.get("cat") == "cpu_op"
        ]
        launches = [
            candidate
            for candidate in candidates
            if candidate.get("cat") in ("cuda_runtime", "cuda_driver")
        ]
        candidates = semantic or launches or candidates
        owners = {}
        for candidate in candidates:
            thread = str(candidate.get("pid")), str(candidate.get("tid"))
            enclosing = [
                span
                for span in by_thread[thread]
                if span["ts"] <= candidate["ts"] < span["ts"] + span["dur"]
            ]
            named = [span for span in enclosing if annotation(span["name"])[1]]
            if named:
                owner = min(named, key=lambda span: span["dur"])
                owners[owner["_index"]] = owner
            elif enclosing and kernel_operation(str(row.get("name", ""))):
                parent = min(enclosing, key=lambda span: span["dur"])
                layer, _ = annotation(parent["name"])
                operation = kernel_operation(row["name"])
                owner = dict(
                    candidate,
                    name=f"tokensim.layer.{layer}.{operation}",
                    _owner_annotation=parent["name"],
                    _operation_evidence="kernel_name",
                )
                owners[owner["_index"]] = owner
        if len(owners) != 1:
            unknown.append(
                Span(
                    id=f"observed:{file_index}:{row['_index']}",
                    operation=canonical(row.get("name", "unknown"), aliases),
                    rank=rank,
                    forward=None,
                    layer=None,
                    phase=None,
                    start_us=row["ts"],
                    end_us=row["ts"] + row["dur"],
                    category=str(row.get("cat", "kernel")),
                    source=str(path),
                    clock=f"profile:{file_index}",
                    kernel_sum_us=row["dur"],
                    busy_us=row["dur"],
                    details={
                        "attribution": "ambiguous" if owners else "unattributed",
                        "name": row.get("name"),
                    },
                )
            )
            continue
        owner = next(iter(owners.values()))
        groups[owner["_index"]].append((row, owner))
    spans = []
    for members in groups.values():
        owner = members[0][1]
        layer, operation = annotation(owner["name"])
        forward = bisect_right(roots, owner["ts"]) - 1 if roots else None
        if forward is not None and forward < 0:
            forward = None
        phase = (
            ("prefill" if forward == 0 else "decode")
            if forward is not None and manifest
            else None
        )
        intervals = [(row["ts"], row["ts"] + row["dur"]) for row, _ in members]
        spans.append(
            Span(
                id=f"observed:{file_index}:scope:{owner['_index']}",
                operation=canonical(operation, aliases),
                rank=rank,
                forward=forward,
                layer=layer,
                phase=phase,
                start_us=min(start for start, _ in intervals),
                end_us=max(end for _, end in intervals),
                category=(
                    "collective"
                    if "reduce" in operation or "gather" in operation
                    else "compute"
                ),
                source=str(path),
                clock=f"profile:{file_index}",
                members=tuple(
                    f"observed:{file_index}:{row['_index']}" for row, _ in members
                ),
                kernel_sum_us=sum(row["dur"] for row, _ in members),
                busy_us=union_duration(intervals),
                details={
                    "attribution": "annotation_and_correlation",
                    "annotation": owner.get("_owner_annotation", owner["name"]),
                    "operation_evidence": owner.get(
                        "_operation_evidence", "module_annotation"
                    ),
                    "service_intervals": intervals,
                    "compute_intervals": [
                        (row["ts"], row["ts"] + row["dur"])
                        for row, _ in members
                        if row.get("cat") != "gpu_memcpy"
                        and kernel_operation(row.get("name", "")) != "all_reduce"
                    ],
                    "communication_intervals": [
                        (row["ts"], row["ts"] + row["dur"])
                        for row, _ in members
                        if row.get("cat") == "gpu_memcpy"
                        or kernel_operation(row.get("name", "")) == "all_reduce"
                    ],
                    "kernel_names": [row.get("name") for row, _ in members],
                },
            )
        )
        fusion = fused_members(spans[-1].operation, spans[-1].details["kernel_names"])
        if fusion:
            spans[-1].details["simulated_members"] = fusion
        if (
            "batch_size" in manifest
            and "input_tokens" in manifest
            and forward is not None
        ):
            spans[-1].shape["forward_tokens"] = manifest["batch_size"] * (
                manifest["input_tokens"] if forward == 0 else 1
            )
    return spans + unknown


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


def read_trace(paths, *, simulated=False, aliases=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    spans, inputs, metadata, diagnostics = [], [], {}, {}
    for index, path in enumerate(paths):
        raw, trace_metadata, digest = _read(path)
        rows, ignored = _complete(raw)
        manifest = {} if simulated else _manifest(path)
        current = (
            _simulated(path, rows, aliases)
            if simulated
            else _observed(path, rows, aliases, index, manifest)
        )
        spans.extend(current)
        inputs.append(dict(path=str(path), sha256=digest, manifest=manifest))
        metadata.update(trace_metadata)
        diagnostics[str(path)] = dict(ignored_events=ignored, device_spans=len(current))
    return ImportedTrace(spans, metadata, inputs, diagnostics)
