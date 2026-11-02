"""Probe a separate LP30 embedding/logits pool with native transformer stages."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer, number
from TokenSim.config.kv_storage import GLOBAL_KV_LAYOUTS
from TokenSim.errors import ConfigurationError
from scripts.lpx.fidelity import load
from scripts.lpx.io_pool import pooled_program, transformer_profile
from scripts.lpx.mfu_metrics import throughput_bounds
from scripts.lpx.mfu_replay import replay_cohorts
from scripts.lpx.replay_profiles import save_profile
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
    parser.add_argument("--resident-requests", type=int)
    parser.add_argument(
        "--global-kv-layout", choices=GLOBAL_KV_LAYOUTS, default="separate"
    )
    parser.add_argument("--shared-kv-kernel", choices=("assemble", "split_dot"))
    parser.add_argument("--issue-us", type=float)
    parser.add_argument("--warmup-rounds", type=int, default=8)
    parser.add_argument("--measured-rounds", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        for name in ("tp", "kvp", "pp", "pool_chips", "warmup_rounds"):
            integer(getattr(args, name), name, minimum=1)
        integer(args.measured_rounds, "measured rounds", minimum=2)
        build_cache()
        config = article_config(load(args.config))
        config["compute"]["attention_mfu"] = 0.5
        if args.shared_kv_kernel is not None:
            config["operator"]["shared_kv_kernel"] = args.shared_kv_kernel
        hardware, model, transformer, admission = transformer_profile(
            config,
            tp=args.tp,
            kvp=args.kvp,
            pp=args.pp,
            queries=1,
            cache=args.output / "profiles",
            resident=args.resident_requests,
            global_kv_layout=args.global_kv_layout,
        )
        profile, pool = pooled_program(
            hardware,
            model,
            transformer,
            tp=args.tp,
            pp=args.pp,
            pool_chips=args.pool_chips,
        )
        profile = save_profile(
            args.output / "program.json.gz",
            profile,
            dict(
                native_component=admission["native_profile"]["input_sha256"],
                program_sha256=profile.metadata["program_sha256"],
            ),
        )
        profile.verify()
        resident = admission["resident_requests"]
        issue_s = (
            args.issue_us * 1e-6
            if args.issue_us is not None
            else profile.metadata["modeled_duration_s"] / resident * 1.01
        )
        number(issue_s, "pool issue interval", positive=True)
        print(
            json.dumps(
                dict(
                    stage="captured",
                    resident=resident,
                    isolated_us=profile.metadata["modeled_duration_s"] * 1e6,
                    issue_us=issue_s * 1e6,
                )
            ),
            flush=True,
        )
        replay = replay_cohorts(
            profile,
            resident_requests=resident,
            initial_spacing_s=issue_s,
            issue_interval_s=issue_s,
            warmup_rounds=args.warmup_rounds,
            measured_rounds=args.measured_rounds,
        )
        chips = args.tp * args.pp + args.pool_chips
        peak = hardware.device("groq_lp30").peak_compute_for("fp8").value * chips
        work = pool["target_useful_work"]["total"]
        bounds = throughput_bounds(profile.nodes, resident_requests=resident, batch=1)
        result = dict(
            config=config,
            layout=dict(
                tp=args.tp, kvp=args.kvp, pp=args.pp, pool_chips=args.pool_chips
            ),
            chips=chips,
            native_transformer=admission,
            pool=pool,
            program=profile.metadata,
            replay=replay,
            mfu_percent=100 * replay["aggregate_tps"] * work / peak,
            bounds={
                **bounds,
                "mfu_percent": 100 * bounds["aggregate_tps"] * work / peak,
            },
            source_cpp_sha256=hashlib.sha256(
                (ROOT / "scripts/lpx/replay_engine.cpp").read_bytes()
            ).hexdigest(),
            scope="Single-token greedy decode with separate shared embedding/LM-head storage, explicit native boundary replacement, arena lifetimes and uncalibrated operator/routes; physical bank placement and compiler execution unverified.",
        )
        if args.global_kv_layout != "separate":
            result[
                "scope"
            ] += " Experimental value-plus-rotary-key FP8 storage uses two FP32 scales per KV head and token; model quality and matching-cache 100K acceptance are unverified."
        write_json(args.output / "result.json", result)
        print(
            json.dumps(
                dict(
                    chips=chips,
                    resident=resident,
                    mfu_percent=result["mfu_percent"],
                    tps=replay["aggregate_tps"],
                    bound_mfu_percent=result["bounds"]["mfu_percent"],
                )
            ),
            flush=True,
        )
        return 0
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"LPX I/O pool: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
