"""Timing accounting and atomic comparison artifact publication."""

import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from TokenSim.profiler.records import union_duration, intervals_for as _intervals


def _accounting(spans):
    intervals = _intervals(spans)
    compute = union_duration(_intervals(spans, {"compute", "kernel", "gpu_kernel"}))
    communication = union_duration(
        _intervals(spans, {"collective", "transfer", "gpu_memcpy"})
    )
    both = union_duration(
        _intervals(
            spans,
            {"compute", "kernel", "gpu_kernel", "collective", "transfer", "gpu_memcpy"},
        )
    )
    return dict(
        makespan_us=max((end for _, end in intervals), default=0)
        - min((start for start, _ in intervals), default=0),
        busy_us=union_duration(
            _intervals(
                spans,
                {
                    "compute",
                    "kernel",
                    "gpu_kernel",
                    "collective",
                    "transfer",
                    "gpu_memcpy",
                    "gpu_memset",
                    "control_device",
                },
            )
        ),
        compute_busy_us=compute,
        communication_busy_us=communication,
        compute_communication_overlap_us=max(0, compute + communication - both),
    )


def _publish(path, writer):
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_comparison(
    simulated, observed, pairs, unmatched, excluded, output, *, mapping_hash
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for left, right in sorted(pairs, key=lambda pair: pair[1].start_us):
        if right.clock not in anchors:
            anchors[right.clock] = dict(
                simulated_id=left.id,
                profiled_id=right.id,
                translation_us=left.start_us - right.start_us,
            )
    matches = []
    for left, right in pairs:
        error = left.duration_us - right.duration_us
        shift = anchors[right.clock]["translation_us"]
        matches.append(
            dict(
                simulated=left.as_dict(),
                profiled=right.as_dict(),
                service_error_us=error,
                absolute_service_error_us=abs(error),
                relative_service_error=(
                    error / right.duration_us if right.duration_us else None
                ),
                relative_error_unavailable_reason=(
                    None if right.duration_us else "zero_observed_duration"
                ),
                start_offset_us=left.start_us - (right.start_us + shift),
                end_offset_us=left.end_us - (right.end_us + shift),
                evidence="rank_forward_layer_order_and_operation",
                duration_scope="invocation_span",
            )
        )
    clocks = sorted({span.clock for span in observed.spans})
    ranks = sorted({span.rank for span in simulated.spans})
    simulated_by_rank = {
        rank: _accounting([span for span in simulated.spans if span.rank == rank])
        for rank in ranks
    }
    report = dict(
        schema_version=1,
        status="compared" if pairs else "no_matches",
        simulated_inputs=simulated.inputs,
        profiled_inputs=observed.inputs,
        mapping_sha256=mapping_hash,
        alignment=dict(
            anchors=anchors,
            method="translation_at_earliest_unambiguous_match_per_clock",
            cross_rank_overlap_available=len(clocks) <= 1,
        ),
        coverage=dict(
            matched_units=len(matches),
            unmatched_units=len(unmatched),
            ambiguous_units=sum(item["status"] == "ambiguous" for item in unmatched),
            excluded_units=len(excluded),
            matched_simulated_service_us=sum(left.duration_us for left, _ in pairs),
            matched_profiled_service_us=sum(right.duration_us for _, right in pairs),
        ),
        accounting=dict(
            simulated=dict(
                makespan_us=_accounting(simulated.spans)["makespan_us"],
                by_rank=simulated_by_rank,
                cumulative_device_busy_us=sum(
                    row["busy_us"] for row in simulated_by_rank.values()
                ),
            ),
            profiled_by_clock={
                clock: _accounting(
                    [span for span in observed.spans if span.clock == clock]
                )
                for clock in clocks
            },
        ),
        diagnostics=dict(
            simulated=simulated.diagnostics, profiled=observed.diagnostics
        ),
        matches=matches,
        unmatched=unmatched,
        excluded=excluded,
    )
    tracks = {}
    events = []
    translated = []
    for side, spans in (("simulated", simulated.spans), ("profiled", observed.spans)):
        for span in spans:
            shift = (
                anchors.get(span.clock, {}).get("translation_us", 0)
                if side == "profiled"
                else 0
            )
            if side == "profiled" and span.clock not in anchors:
                shift = -min(
                    (item.start_us for item in spans if item.clock == span.clock),
                    default=0,
                )
            translated.append((side, span, span.start_us + shift))
    origin = min((start for _, _, start in translated), default=0)
    for side, span, start in translated:
        key = side, span.rank, span.clock
        if key not in tracks:
            tracks[key] = len(tracks) + 1
            events.append(
                dict(
                    ph="M",
                    name="thread_name",
                    pid=1,
                    tid=tracks[key],
                    args={"name": f"{side}:rank:{span.rank}:{span.clock}"},
                )
            )
        events.append(
            dict(
                ph="X",
                name=span.operation,
                cat=span.category,
                pid=1,
                tid=tracks[key],
                ts=start - origin,
                dur=span.duration_us,
                args=dict(span.as_dict(), comparison_side=side),
            )
        )
    timeline = dict(
        displayTimeUnit="us",
        traceEvents=events,
        metadata=dict(
            schema_version=1, alignment=report["alignment"], display_origin_us=origin
        ),
    )
    _publish(
        output / "comparison.trace.json",
        lambda stream: json.dump(timeline, stream, allow_nan=False),
    )

    def write_csv(stream):
        writer = csv.writer(stream)
        writer.writerow(
            [
                "status",
                "side",
                "operation",
                "rank",
                "forward",
                "layer",
                "simulated_id",
                "profiled_id",
                "simulated_us",
                "profiled_us",
                "service_error_us",
                "absolute_error_us",
                "relative_error",
            ]
        )
        for row in matches:
            left, right = row["simulated"], row["profiled"]
            writer.writerow(
                [
                    "matched",
                    "both",
                    left["operation"],
                    left["rank"],
                    left["forward"],
                    left["layer"],
                    left["id"],
                    right["id"],
                    left["duration_us"],
                    right["duration_us"],
                    row["service_error_us"],
                    row["absolute_service_error_us"],
                    row["relative_service_error"],
                ]
            )
        for row in unmatched + excluded:
            writer.writerow(
                [
                    row.get("status", "excluded"),
                    row["side"],
                    row["operation"],
                    row["rank"],
                    row["forward"],
                    row["layer"],
                    row["id"] if row["side"] == "simulated" else "",
                    row["id"] if row["side"] == "profiled" else "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    _publish(output / "operators.csv", write_csv)
    _publish(
        output / "comparison.json",
        lambda stream: json.dump(report, stream, indent=2, allow_nan=False),
    )
    return report
