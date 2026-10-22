"""Reconcile isolated operator composition with the native full-model DAG.

Run from the repository root with .venv/bin/python scripts/audit_lpx_latency.py.
Free-service variants attribute costs and do not implement valid new layouts.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TokenSim.af.providers.traffic import TransferPlanMixin
from TokenSim.af.providers.transfer import TransferEstimateMixin
from TokenSim.hardware import HardwareContext
from scripts.lpx.latency_ledger import (
    build,
    catalog_collective,
    close,
    composed,
    free_services,
    identity,
    native,
)

VARIANTS = ("default", "catalog", "free_handoffs", "free_handoffs_and_gathers")
LENGTHS = ((10000, 1500), (100000, 2000))


def patched(runtime, variant):
    stack = ExitStack()
    if variant != "default":
        stack.enter_context(
            patch.object(TransferPlanMixin, "_collective", catalog_collective(runtime))
        )
    if variant.startswith("free_"):
        stack.enter_context(
            patch.object(
                TransferEstimateMixin,
                "estimate",
                free_services(
                    handoffs=True,
                    residual_gathers=variant == "free_handoffs_and_gathers",
                ),
            )
        )
    return stack


def validate_interval(measure, inputs, outputs, samples):
    start, end = inputs + 1, inputs + outputs - 1
    contexts = [
        round(start + (end - start) * i / (samples - 1)) for i in range(samples)
    ]
    points = [measure(context) for context in contexts]
    low, high = points[0]["latency_us"], points[-1]["latency_us"]
    errors = [
        abs(
            point["latency_us"]
            - (low + (high - low) * (point["context"] - start) / (end - start))
        )
        for point in points
    ]
    midpoint = measure((start + end) // 2)
    close(midpoint["latency_us"], (low + high) / 2)
    if max(errors) > 1e-7:
        raise AssertionError("context interval is not affine within audit tolerance")
    return dict(
        input_tokens=inputs,
        output_tokens=outputs,
        generation_intervals=outputs - 1,
        contexts=contexts,
        sample_latencies_us=[point["latency_us"] for point in points],
        max_affine_error_us=max(errors),
        mean_latency_us=(low + high) / 2,
        estimated_generation_tps=1e6 / ((low + high) / 2),
        midpoint=midpoint,
    )


def operator_difference(left, right):
    a = {identity(row["table"], row["key"]): row for row in left["operators"]}
    b = {identity(row["table"], row["key"]): row for row in right["operators"]}
    rows = []
    for key in sorted(a.keys() | b.keys()):
        count = b.get(key, {}).get("count", 0) - a.get(key, {}).get("count", 0)
        if count:
            row = b.get(key, a.get(key))
            rows.append(
                dict(
                    table=row["table"],
                    key=row["key"],
                    native_minus_composed_count=count,
                    native_minus_composed_us=count * row["latency_us"],
                )
            )
    close(
        sum(row["native_minus_composed_us"] for row in rows),
        right["operator_budget"]["service_us"] - left["operator_budget"]["service_us"],
    )
    return rows


def bridge(isolated, points):
    default, catalog, no_handoffs, no_gathers = [points[name] for name in VARIANTS]
    parts = dict(
        collective_algorithm_us=default["latency_us"] - catalog["latency_us"],
        internal_af_copy_and_queue_us=catalog["latency_us"] - no_handoffs["latency_us"],
        residual_gathers_us=no_handoffs["latency_us"] - no_gathers["latency_us"],
        boundaries_and_residual_shape_us=no_gathers["latency_us"]
        - isolated["latency_us"],
    )
    close(sum(parts.values()), default["latency_us"] - isolated["latency_us"])
    boundaries = Counter()
    for node in no_gathers["path"]:
        if node["layer"] == -1:
            boundaries[node["operation"]] += node["service_us"]
        boundaries["host_readiness_gap"] += node["readiness_gap_us"]
    # Final norm and LM head are present in both paths.
    for key in ("final_norm", "lm_head"):
        boundaries.pop(key)
    differences = operator_difference(isolated, default)
    boundaries["residual_shape_difference"] = sum(
        row["native_minus_composed_us"]
        for row in differences
        if row["key"].get("op_name") == "residual_add"
    )
    close(sum(boundaries.values()), parts["boundaries_and_residual_shape_us"])
    return dict(
        differences_us=parts,
        boundary_difference_us=dict(boundaries),
        operator_multiset_difference=differences,
    )


def source_manifest():
    paths = [
        "scripts/audit_lpx_latency.py",
        "scripts/lpx/latency_ledger.py",
        "scripts/lpx/native_config.py",
        "scripts/lpx/aa.py",
        "TokenSim/af/serving/prefill.py",
        "TokenSim/af/serving/service.py",
        "TokenSim/af/execution/model.py",
        "TokenSim/af/planning/copies.py",
        "TokenSim/af/providers/traffic.py",
        "TokenSim/af/providers/components.py",
        "TokenSim/af/providers/attention.py",
        "TokenSim/latency/operator_table/composition.py",
        "TokenSim/latency/operator_table/layers.py",
        "TokenSim/operator_data/analytical/groq.py",
        "TokenSim/comm/collectives.py",
        "data/devices/groq_lp30.yaml",
        "data/topologies/groq_lpx256.yaml",
        "data/topologies/links.yaml",
        "data/models/gemma-4-31b-text.yaml",
    ]
    return dict(
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-latency-ledger"
    )
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    if args.samples < 3 or args.samples % 2 != 1:
        parser.error("--samples must be an odd integer of at least three")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    hardware = HardwareContext.load(ROOT / "data")
    runtime = build(hardware)
    output = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="isolated decode accounting; no prefill or 50-request run",
        model="gemma-4-31b-text",
        layout=dict(lpu_count=256, tp=16, pp=16, batch=1),
        precision=dict(weights="fp8", kv="fp8", activation="bf16"),
        assumptions=dict(hardware.device("groq_lp30").analytical),
        sources=source_manifest(),
        composed=[
            validate_interval(lambda c: composed(runtime, c), i, o, args.samples)
            for i, o in LENGTHS
        ],
        native={},
    )
    for variant in VARIANTS:
        runtime = build(hardware)
        with patched(runtime, variant):
            rows = [
                validate_interval(
                    lambda c: native(runtime, c), inputs, outputs, args.samples
                )
                for inputs, outputs in LENGTHS
            ]
            if variant in {"default", "catalog"}:
                for row in rows:
                    start = row["input_tokens"] + 1
                    contiguous = native(runtime, start, steps=8, details=False)
                    isolated = [
                        native(runtime, start + i, details=False)["latency_us"]
                        for i in range(8)
                    ]
                    error = max(
                        abs(a - b)
                        for a, b in zip(contiguous["step_latencies_us"], isolated)
                    )
                    if error > 1e-7:
                        raise AssertionError(
                            "contiguous replay differs from isolated steps"
                        )
                    row["contiguous_8_step_error_us"] = error
            output["native"][variant] = rows
        print(
            variant,
            [round(row["estimated_generation_tps"], 2) for row in rows],
            flush=True,
        )
    output["reconciliation"] = [
        bridge(
            output["composed"][index]["midpoint"],
            {name: output["native"][name][index]["midpoint"] for name in VARIANTS},
        )
        for index in range(len(LENGTHS))
    ]
    output["host_runtime_s"] = perf_counter() - started
    (args.output / "ledger.json").write_text(json.dumps(output, indent=2) + "\n")
    print("saved", args.output / "ledger.json", flush=True)


if __name__ == "__main__":
    main()
