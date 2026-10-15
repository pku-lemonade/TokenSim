"""Export native LPX measurements and compact evidence from audited summaries."""

import csv
import hashlib
import json
from pathlib import Path


POINT_FIELDS = (
    "case",
    "mode",
    "batch",
    "status",
    "tokens_per_s_per_user",
    "output_tokens_per_s",
    "tokens_per_s_per_mw",
    "normalized_throughput",
    "power_watts",
    "reason",
)


def write_points_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=POINT_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _point_evidence(row):
    result = {key: row[key] for key in POINT_FIELDS if key in row}
    if row["status"] != "ok":
        return result
    metric = row["steady_state"]
    result.update(
        inventory=row["inventory"],
        timings=row["timings"],
        steady_state={
            key: value for key, value in metric.items() if not isinstance(value, dict)
        },
        maximum_resource_utilization=max(
            metric["resource_utilization"].values(), default=0
        ),
        total_resource_queue_s=sum(metric["resource_queue_s"].values()),
        turns=row["turns"],
        handoff_bytes_all_phases=sum(e["bytes"] for e in row["handoff_events"]),
        control_bytes_all_phases=sum(e["bytes"] for e in row["control_events"]),
    )
    cycles = [
        event
        for event in row["verification_events"]
        if not event["warmup"] and metric["start_s"] <= event["time"] <= metric["end_s"]
    ]
    if row["mode"] == "draft":
        result["measured_verification"] = {
            "cycles": len(cycles),
            **{
                key: sum(
                    event[key]
                    * min(
                        row["micro_batch"],
                        row["clients"] - event["cohort"] * row["micro_batch"],
                    )
                    for event in cycles
                )
                for key in ("committed_tokens", "draft_catchup_tokens")
            },
        }
    return result


def write_evidence_bundle(output, payload, audit, *, source_bytes, source_name):
    """Keep source bytes immutable and identify them with a SHA-256 digest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = (
        "executor",
        "git_revision",
        "git_worktree_dirty",
        "end_git_revision",
        "end_git_worktree_dirty",
        "workload",
        "model",
        "draft_model",
        "architecture",
        "run_settings",
        "cases",
        "hardware",
        "power_overrides",
        "facility_watts",
        "normalization_tokens_per_s_per_mw",
        "frontiers",
        "limitations",
    )
    evidence = {
        "schema": "tokensim.lpx_evidence",
        "version": 1,
        "source_summary": {
            "name": source_name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "audit": audit,
        "power_model": audit["power_model"],
        **{key: payload[key] for key in keys if key in payload},
        "results": [_point_evidence(row) for row in payload["results"]],
    }
    path = output / "evidence.json"
    path.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_points_csv(output / "points.csv", payload["results"])
    return path
