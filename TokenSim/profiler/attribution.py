"""Associate device activities with observed CPU module ownership."""

from bisect import bisect_right
from collections import defaultdict
import re
from pathlib import Path

from TokenSim.profiler.annotations import kernel_operation, fused_members
from TokenSim.profiler.records import Span, annotation, canonical, union_duration


def activity_details(rows):
    """Keep device work and collective residence separate from module gaps."""
    intervals = [(row["ts"], row["ts"] + row["dur"]) for row in rows]
    collectives = [
        kernel_operation(row.get("name", ""))
        in {"all_reduce", "all_gather", "reduce_scatter"}
        for row in rows
    ]
    return dict(
        service_intervals=intervals,
        compute_intervals=[
            interval
            for row, interval, collective in zip(rows, intervals, collectives)
            if row.get("cat") != "gpu_memcpy" and not collective
        ],
        communication_intervals=[
            interval
            for row, interval, collective in zip(rows, intervals, collectives)
            if row.get("cat") == "gpu_memcpy" or collective
        ],
        collective_calls=[
            dict(
                operation=kernel_operation(row["name"]),
                wait_model=(
                    "peer_start_barrier"
                    if any(
                        name in row["name"].lower()
                        for name in (
                            "cross_device_reduce_1stage",
                            "cross_device_reduce_2stage",
                        )
                    )
                    else "unresolved"
                ),
                start_us=interval[0],
                end_us=interval[1],
                kernel_index=row["_index"],
            )
            for row, interval, collective in zip(rows, intervals, collectives)
            if collective
        ],
        kernel_names=[row.get("name") for row in rows],
    )


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
                        **activity_details([row]),
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
                    **activity_details([row for row, _ in members]),
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
