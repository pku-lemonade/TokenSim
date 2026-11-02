"""Replay complete prompt and output lifecycles on admitted LP30 pipelines."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.kv_storage import GLOBAL_KV_LAYOUTS
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.fidelity import load
from scripts.lpx.serving_bank import NativeServingBank
from scripts.lpx.serving_metrics import serving_work
from scripts.lpx.serving_pipeline import (
    ServingPipeline,
    ServingSchedule,
    ServingWorkload,
)
from scripts.run_lpx_article_sweep import article_config
from scripts.search_lpx_mfu import build_cache
from scripts.search_lpx_native import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--kvp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=10)
    parser.add_argument("--pool-chips", type=int, default=3)
    parser.add_argument("--inputs", type=int, default=100000)
    parser.add_argument("--outputs", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--context-bands", type=int, default=4)
    parser.add_argument("--prefill-quantum", type=int, default=1)
    parser.add_argument("--decode-priority", action="store_true")
    parser.add_argument(
        "--global-kv-layout", choices=GLOBAL_KV_LAYOUTS, default="separate"
    )
    args = parser.parse_args(argv)
    try:
        started = perf_counter()
        build_cache()
        config = article_config(load(args.config))
        config["compute"]["attention_mfu"] = 0.5
        workload = ServingWorkload(
            args.inputs, args.outputs, args.chunk, args.clients, args.cycles
        )
        schedule = ServingSchedule(args.prefill_quantum, args.decode_priority)
        bank = NativeServingBank(
            config,
            workload,
            tp=args.tp,
            kvp=args.kvp,
            pp=args.pp,
            pool_chips=args.pool_chips,
            cache=args.output / "profiles",
            bands=args.context_bands,
            global_kv_layout=args.global_kv_layout,
        )
        print(
            json.dumps(
                dict(
                    stage="admitted",
                    clients=args.clients,
                    transformer_headroom=bank.admission["memory"][
                        "minimum_headroom_bytes"
                    ],
                    pool_headroom=min(x["headroom_bytes"] for x in bank.pool_memory),
                )
            ),
            flush=True,
        )
        replay = ServingPipeline(bank, workload, schedule).run()
        work = serving_work(bank.model, inputs=args.inputs, outputs=args.outputs)
        peak = (
            bank.chips * bank.hardware.device("groq_lp30").peak_compute_for("fp8").value
        )
        result = dict(
            config=config,
            workload=asdict(workload),
            schedule=asdict(schedule),
            bank=bank.describe(),
            work_per_request=work,
            replay=replay,
            allocated_fp8_peak=peak,
            end_to_end_mfu_percent=100
            * work["total"]
            * replay["requests"]
            / (replay["seconds"] * peak),
            decode_work_fraction=work["decode"]["total"] / work["total"],
            host_seconds=perf_counter() - started,
            scope="Conservative complete-request operator replay with exact useful-work credit. Prefill and decode share one allocation; whole-buffer collectives, explicit stage gates and context reservations are modeled. This MFU includes prompt work and is a different metric from resident-prefix decode MFU. Hardware compiler execution, physical banks, model quality, tokenization and external request transport remain unverified.",
        )
        write_json(args.output / "result.json", result)
        print(
            json.dumps(
                dict(
                    output=str(args.output / "result.json"),
                    mfu_percent=result["end_to_end_mfu_percent"],
                    output_tps=replay["output_tps"],
                    requests_per_s=replay["requests_per_s"],
                    seconds=replay["seconds"],
                )
            ),
            flush=True,
        )
        return 0
    except (ConfigurationError, SimulationStateError, OSError, ValueError) as exc:
        print(f"LPX complete serving: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
