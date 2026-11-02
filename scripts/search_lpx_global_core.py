"""Screen bounded concurrent global-core GPU / LP30 decode configurations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import ConfigurationError
from scripts.lpx.fidelity import load
from scripts.lpx.sqd_memory import CoreMemory
from scripts.lpx.sqd_probe import probe
from scripts.lpx.replay_profiles import source_identity
from scripts.run_lpx_article_sweep import article_config
from scripts.search_lpx_mfu import build_cache
from scripts.search_lpx_native import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--lp-tp", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--pp", type=int, nargs="+", default=[6, 8, 10, 12, 15, 20, 30, 60]
    )
    parser.add_argument("--gpu", default="b200_sxm")
    parser.add_argument("--gpu-tp", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--resident",
        type=int,
        default=0,
        help="Zero selects maximum projected capacity.",
    )
    parser.add_argument("--request-limit", type=int, default=256)
    parser.add_argument("--max-cards", type=int, default=512)
    parser.add_argument("--cross-latency-us", type=float, default=2.0)
    parser.add_argument("--cross-bandwidth", type=float, default=200e9)
    parser.add_argument("--host-submit-us", type=float, default=4.0)
    parser.add_argument("--relax-lp-collectives", action="store_true")
    parser.add_argument("--spacing-us", type=float, default=0.0)
    parser.add_argument("--pace-us", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--measured", type=int, default=32)
    parser.add_argument("--bounds-only", action="store_true")
    parser.add_argument("--save-profiles", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_cache()
    config = article_config(load(args.config))
    config["compute"]["attention_mfu"] = 0.5
    if args.max_cards > 256:
        config["fabric"]["inter_rack_us"] = 2.0
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    for lp_tp in args.lp_tp:
        try:
            memory = CoreMemory(config, lp_tp, args.batch)
        except ConfigurationError as exc:
            results.append(dict(status="infeasible", lp_tp=lp_tp, reason=str(exc)))
            continue
        for pp in args.pp:
            for gpu_tp in args.gpu_tp:
                for replicas in args.replicas:
                    if lp_tp * pp + 3 + 10 * gpu_tp * replicas > args.max_cards:
                        continue
                    case = dict(lp_tp=lp_tp, pp=pp, gpu_tp=gpu_tp, replicas=replicas)
                    began = perf_counter()
                    try:
                        key = f"lp{lp_tp}-pp{pp}-g{gpu_tp}-r{replicas}"
                        result = probe(
                            memory,
                            pp=pp,
                            gpu_device=args.gpu,
                            gpu_tp=gpu_tp,
                            replicas=replicas,
                            resident=args.resident or None,
                            request_limit=args.request_limit,
                            warmup=args.warmup,
                            measured=args.measured,
                            bounds_only=args.bounds_only,
                            profile_folder=(
                                args.output / key if args.save_profiles else None
                            ),
                            spacing_us=args.spacing_us,
                            pace_us=args.pace_us,
                            cross_latency_us=args.cross_latency_us,
                            cross_bandwidth=args.cross_bandwidth,
                            host_submit_us=args.host_submit_us,
                            collectives=not args.relax_lp_collectives,
                        )
                    except ConfigurationError as exc:
                        result = dict(status="infeasible", case=case, reason=str(exc))
                    result["host_seconds"] = perf_counter() - began
                    results.append(result)
                    print(
                        json.dumps(
                            dict(
                                case=case,
                                status=result["status"],
                                resident=result.get("lp30_memory", {}).get(
                                    "resident_requests"
                                ),
                                mfu_percent=result.get("mfu_percent"),
                                bound_mfu_percent=result.get("bounds", {}).get(
                                    "mfu_percent"
                                ),
                            )
                        ),
                        flush=True,
                    )
                    write_json(
                        args.output / "results.json",
                        dict(
                            source_sha256=source_identity(),
                            driver_sha256=hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                            config=config,
                            arguments={
                                k: str(v) if isinstance(v, Path) else v
                                for k, v in vars(args).items()
                            },
                            results=results,
                            maximum_certified=False,
                        ),
                    )


if __name__ == "__main__":
    main()
