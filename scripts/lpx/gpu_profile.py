"""Profiler configuration, trace provenance and streamed timing diagnostics."""

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_gpu_smoke import first_difference, read_outputs
from scripts.run_lpx_gpu_followup import validate_counters

ROOT = Path(__file__).resolve().parents[2]


def profiler_config(directory):
    directory = directory.resolve()
    if directory.is_relative_to(ROOT):
        raise WorkloadValidationError(
            "Profiler traces must remain outside the repository"
        )
    return dict(
        profiler="torch",
        torch_profiler_dir=str(directory),
        torch_profiler_activities=["CPU", "CUDA"],
        torch_profiler_with_stack=True,
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_use_gzip=True,
        torch_profiler_dump_cuda_time_total=False,
        detailed_trace_annotation=True,
        ignore_frontend=False,
    )


def file_record(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)


def trace_inventory(directory):
    return [
        file_record(path) for path in sorted(directory.rglob("*")) if path.is_file()
    ]


def read_trace_events(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace.get("traceEvents") if isinstance(trace, dict) else None
    if (
        not isinstance(events, list)
        or not events
        or any(not isinstance(event, dict) for event in events)
    ):
        raise WorkloadValidationError("Expected a nonempty Chrome trace event array")
    return events


def trace_summary(path):
    events = read_trace_events(path)
    categories = Counter(str(event.get("cat", "metadata")) for event in events)
    return dict(
        path=str(path.resolve()),
        event_count=len(events),
        categories=dict(categories),
        kernel_events=sum(
            count
            for category, count in categories.items()
            if "kernel" in category.split(",")
        ),
    )


def stream_step_us(turn):
    tokens = turn["completion_tokens"] - 1
    elapsed = turn["decode_s"]
    if tokens < 1 or not math.isfinite(elapsed) or elapsed <= 0:
        raise WorkloadValidationError(
            "Streamed decode timing needs tokens and positive elapsed time"
        )
    return elapsed / tokens * 1e6


def read_profile_turn(path, context):
    rows = read_outputs(path)
    if len(rows) != 1 or len(rows[0]["turns"]) != 1:
        raise WorkloadValidationError("Profiling expects one request and one turn")
    row = rows[0]
    config = row["config"]
    if (
        row["set"] != "aa-docs"
        or row["meta"]["context"] != context
        or row["thinking"] is not True
        or config["max_tokens_override"] != 256
        or config["weight_precision"] != "bf16"
        or config["kv_precision"] != "bf16"
    ):
        raise WorkloadValidationError(
            "Profiling requires the specified capped BF16 document"
        )
    validate_counters(rows, "plain", 0)
    turn = row["turns"][0]
    if (
        turn["finish_reason"] not in ("length", "stop")
        or not 2 <= turn["completion_tokens"] <= 256
        or len(turn["token_ids"]) != turn["completion_tokens"]
    ):
        raise WorkloadValidationError(
            "Invalid profiled output termination or token count"
        )
    stream_step_us(turn)
    return row, turn


def timing_comparison(control, profiled):
    before = stream_step_us(control)
    after = stream_step_us(profiled)
    difference = first_difference(control["token_ids"], profiled["token_ids"])
    return dict(
        control_stream_step_us=before,
        profiled_stream_step_us=after,
        profiled_over_control=after / before,
        control_completion_tokens=control["completion_tokens"],
        profiled_completion_tokens=profiled["completion_tokens"],
        identical_tokens=difference is None,
        first_different_token=difference,
        note="Client delivery interval per decode token; includes buffering and profiler overhead; not a GPU kernel-time breakdown",
    )


def verify_records(records):
    for expected in records:
        path = Path(expected["path"])
        if not path.is_file() or file_record(path) != expected:
            raise WorkloadValidationError(
                f"Missing or changed profiling artifact: {path}"
            )
