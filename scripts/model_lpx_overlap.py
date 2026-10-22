"""Compare fixed-layout native decode with scheduled LPX output-vector overlap."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.comm.streaming import StreamingConfig
from TokenSim.hardware import HardwareContext
from scripts.lpx.latency_ledger import build
from scripts.lpx.replay_profiles import capture_profile, source_identity
from scripts.lpx.streaming import duration


def evaluate_variants(runtime, context, args):
    result = {}
    for launch in args.collective_launch_us:
        settings = [
            ("catalog", StreamingConfig(overlap=False, collective_launch_us=launch))
        ]
        settings += [
            (
                f"phase-only-v{vector}",
                StreamingConfig(
                    vector_bytes=vector, early_output=False, collective_launch_us=launch
                ),
            )
            for vector in args.vector_bytes
        ]
        settings += [
            (
                f"overlap-v{vector}",
                StreamingConfig(vector_bytes=vector, collective_launch_us=launch),
            )
            for vector in args.vector_bytes
        ]
        for label, config in settings:
            profile = capture_profile(
                runtime,
                "ledger",
                batch=1,
                queries=1,
                phase="decode",
                emit_logits=True,
                context=context,
                dp_rank=0,
                seed=42,
                sample=0,
                streaming=asdict(config),
            )
            result[f"{label}-launch{launch}"] = dict(
                latency_us=profile.metadata["modeled_duration_s"] * 1e6,
                native_us=profile.metadata["native_duration_s"] * 1e6,
                **profile.metadata["streaming"],
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-streaming-model"
    )
    parser.add_argument(
        "--vector-bytes", type=int, nargs="+", default=[320, 640, 10752]
    )
    parser.add_argument("--collective-launch-us", type=float, nargs="+", default=[0.5])
    parser.add_argument("--context-samples", type=int, default=3)
    parser.add_argument("--colocated-dense-reuse", action="store_true")
    args = parser.parse_args(argv)
    from TokenSim.config.validation import integer

    integer(args.context_samples, "context samples", minimum=2)
    for vector in args.vector_bytes:
        StreamingConfig(vector_bytes=vector)
    for launch in args.collective_launch_us:
        StreamingConfig(collective_launch_us=launch)
    runtime = build(
        HardwareContext.load(ROOT / "data"),
        colocated_dense_reuse=args.colocated_dense_reuse,
    )
    rows = []
    for inputs, outputs in ((10000, 1500), (100000, 2000)):
        low, high = inputs + 1, inputs + outputs - 1
        contexts = [
            round(low + (high - low) * i / (args.context_samples - 1))
            for i in range(args.context_samples)
        ]
        for context in contexts:
            started = perf_counter()
            native = capture_profile(
                runtime,
                "ledger",
                batch=1,
                queries=1,
                phase="decode",
                emit_logits=True,
                context=context,
                dp_rank=0,
                seed=42,
                sample=0,
            )
            result = evaluate_variants(runtime, context, args)
            row = dict(
                input_tokens=inputs,
                output_tokens=outputs,
                context=context,
                native_us=duration(native.nodes) * 1e6,
                variants=result,
                host_seconds=perf_counter() - started,
            )
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        context=context,
                        latency_us={
                            label: value["latency_us"]
                            for label, value in result.items()
                        },
                    )
                ),
                flush=True,
            )
    summary = []
    for inputs, outputs in ((10000, 1500), (100000, 2000)):
        group = [row for row in rows if row["input_tokens"] == inputs]
        for label in group[0]["variants"]:
            values = [row["variants"][label]["latency_us"] for row in group]
            error = max(
                abs(
                    value
                    - (
                        values[0]
                        + (values[-1] - values[0])
                        * (row["context"] - group[0]["context"])
                        / (group[-1]["context"] - group[0]["context"])
                    )
                )
                for row, value in zip(group, values)
            )
            if error > 1e-7:
                raise AssertionError(
                    "context sweep is not affine; use full interval integration"
                )
            latency = (values[0] + values[-1]) / 2
            summary.append(
                dict(
                    input_tokens=inputs,
                    output_tokens=outputs,
                    variant=label,
                    mean_latency_us=latency,
                    generation_tps=1e6 / latency,
                    max_affine_error_us=error,
                )
            )
    source_files = [
        ROOT / "scripts/model_lpx_overlap.py",
        *[
            ROOT / name
            for name in ("TokenSim/comm/streaming.py", "scripts/lpx/streaming.py")
        ],
    ]
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        method="native_decode_dag_refinement_context_sweep",
        scope="Single-client generation intervals; TTFT excluded; no measured 50-request replay.",
        layout=dict(
            lp30=256,
            tp=16,
            pp=16,
            weight_dtype="fp8",
            kv_dtype="fp8",
            activation_dtype="bf16",
            colocated_dense_reuse=args.colocated_dense_reuse,
        ),
        source_sha256=source_identity(),
        files={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        rows=rows,
        summary=summary,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
