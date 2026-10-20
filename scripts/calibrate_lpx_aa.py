"""Run conditional AA serial calibration with explicit deployment hypotheses."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from multiprocessing import get_context
from unittest.mock import patch

from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext, DeviceCatalog, LinkCatalog
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.utils import BoundedCache
from scripts.lpx.aa import scaled_efficiency, serial
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration
from scripts.reproduce_lpx import repository_state, write_json

TARGETS = ((10000, 1500, 3382), (100000, 2000, 3431))


def communication_hypothesis(hardware, latency_us=None, launch_us=None):
    if latency_us is not None:
        number(latency_us, "C2C latency microseconds")
        hardware = replace(
            hardware,
            links=LinkCatalog(
                [
                    (
                        replace(
                            link,
                            latency_us=latency_us,
                            source_id="assumption:aa-communication-variant",
                            grade="D",
                        )
                        if link.link_id in {"groq_lp30_injection", "groq_lp30_c2c_112g"}
                        else link
                    )
                    for link in hardware.links
                ]
            ),
        )
    if launch_us is not None:
        number(launch_us, "collective launch microseconds")
        hardware = replace(
            hardware,
            devices=DeviceCatalog(
                [
                    (
                        replace(
                            device,
                            analytical={
                                **device.analytical,
                                "collective_launch_us": launch_us,
                            },
                        )
                        if device.device_id == "groq_lp30"
                        else device
                    )
                    for device in hardware.devices
                ]
            ),
        )
    return hardware


@contextmanager
def dense_step_cache():
    """Memoize pure isolated dense costs; SimPy still executes every token.

    This process-local scope excludes stochastic MoE and compressed modules.
    Operator counters describe unique evaluated geometries while request
    timestamps and admission remain fully executed.
    """
    original = OperatorTableLatencyBackend.estimate_step_latency

    def estimate(backend, requests):
        if backend.model.is_moe or backend.model.components or len(requests) != 1:
            raise ConfigurationError(
                "AA step caching requires one ordinary dense request"
            )
        request = requests[0]
        key = (
            request.prefill_len,
            request.num_computed_tokens,
            request.query_tokens,
            request.generation_idx > 0,
            request.needs_recompute,
            bool(getattr(request, "emit_logits", True)),
        )
        if not hasattr(backend, "_aa_step_cache"):
            backend._aa_step_cache = BoundedCache(8192)
        cached = backend._aa_step_cache.get(key)
        if cached is None:
            cached = original(backend, requests)
            backend._aa_step_cache[key] = cached
        return cached

    with patch.object(OperatorTableLatencyBackend, "estimate_step_latency", estimate):
        yield


def evaluate_candidate(parameters):
    options = dict(parameters)
    output = options.pop("output")
    resume = options.pop("resume")
    hardware = HardwareContext.load(ROOT / "data")
    hardware = communication_hypothesis(
        hardware, options.get("c2c_latency_us"), options.get("collective_launch_us")
    )
    base = hardware.models.resolve(options["model"])
    options["model_spec"] = base.to_dict()

    def run(directory):
        model = replace(
            base, dtype=options["weight_dtype"], kv_cache_dtype=options["kv_dtype"]
        )
        layout = Layout(**options["layout"])
        lengths = options["lengths"]
        rho_in, rho_out = options["rho_in"], options["rho_out"]
        try:
            runtime = configuration(hardware, model, layout, namespace="aa-check")
            memory = ResidentKV(runtime.plan)
            owner = next(iter(runtime.plan.request_owners))
            maximum = max(round(i * rho_in) + round(o * rho_out) for i, o, _ in lengths)
            memory.admit(
                0,
                owner.domain,
                owner.instance,
                maximum_context=maximum,
                current_context=0,
            )
        except ConfigurationError as exc:
            return dict(status="infeasible", reason=str(exc), hypothesis=options)

        def sample(input_tokens, output_tokens, scale=1, requests=None, warmup=None):
            result = serial(
                scaled_efficiency(hardware, scale),
                model,
                layout,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                requests=options["requests"] if requests is None else requests,
                warmup=options["warmup"] if warmup is None else warmup,
                prefill_chunk=options["prefill_chunk"],
                rho_in=rho_in,
                rho_out=rho_out,
            )
            result["exact_step_cache"] = options.get("step_cache", True)
            if result["exact_step_cache"]:
                result["operator_statistics_scope"] = (
                    "unique evaluated dense step geometries; all requests and token clocks executed"
                )
            return result

        raw = [sample(i, o) for i, o, _ in lengths]
        print(
            f"AA raw {options['name']}: {[round(r['standardized_generation_tps']['p50'], 1) for r in raw]}",
            flush=True,
        )
        result = dict(
            status="ok",
            hypothesis=options,
            model=model.to_dict(),
            prefit=raw,
            efficiency_scale=1.0,
            fit_status="disabled",
        )
        result["communication_hypothesis"] = dict(
            c2c_latency_us=hardware.link("groq_lp30_injection").latency_us,
            collective_launch_us=hardware.device("groq_lp30").analytical[
                "collective_launch_us"
            ],
        )
        selected = raw
        if options["fit"]:
            target = lengths[0][2]
            upper = min(
                1 / hardware.device("groq_lp30").analytical[key]
                for key in ("gemm_mfu", "attention_mfu", "memory_efficiency")
            )
            lower = 0.05
            trials = []
            for scale in (lower, upper):
                measured = sample(*lengths[0][:2], scale, requests=1, warmup=0)
                trials.append((scale, measured["standardized_generation_tps"]["p50"]))
            if trials[0][1] <= target <= trials[1][1]:
                for _ in range(9):
                    scale = (lower + upper) / 2
                    measured = sample(*lengths[0][:2], scale, requests=1, warmup=0)
                    speed = measured["standardized_generation_tps"]["p50"]
                    trials.append((scale, speed))
                    if speed < target:
                        lower = scale
                    else:
                        upper = scale
                scale = min(trials, key=lambda pair: abs(pair[1] / target - 1))[0]
                result["fit_status"] = "fit_10k_only"
            else:
                scale = min(trials, key=lambda pair: abs(pair[1] / target - 1))[0]
                result["fit_status"] = "target_outside_physical_efficiency_bounds"
            selected = [sample(i, o, scale) for i, o, _ in lengths]
            result.update(
                efficiency_scale=scale,
                fitted=selected,
                fit_trials=trials,
                efficiency_bounds=[
                    0.05,
                    min(
                        1 / hardware.device("groq_lp30").analytical[key]
                        for key in ("gemm_mfu", "attention_mfu", "memory_efficiency")
                    ),
                ],
            )
        errors = [
            abs(row["standardized_generation_tps"]["p50"] / target - 1)
            for row, (_, _, target) in zip(selected, lengths)
        ]
        ratio = (
            selected[1]["standardized_generation_tps"]["p50"]
            / selected[0]["standardized_generation_tps"]["p50"]
        )
        ratio_error = abs(ratio / (lengths[1][2] / lengths[0][2]) - 1)
        result.update(
            relative_errors=errors,
            predicted_context_ratio=ratio,
            context_ratio_relative_error=ratio_error,
            holdout_passed=errors[1] <= 0.1 and ratio_error <= 0.05,
            evidence_scope="published-workload-hypothesis; identical deterministic requests; token ratios and serving layout unverified",
        )
        return result

    runner = ExperimentRunner(
        Path(output) / ".experiments",
        inputs=(ROOT / "data",)
        + ((Path(options["model"]),) if Path(options["model"]).is_file() else ()),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
        resume=resume,
    )
    if options.get("step_cache", True):
        with dense_step_cache():
            return runner.run(options, run).result
    return runner.run(options, run).result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-study/aa")
    parser.add_argument("--model", default="gemma-4-31b-text")
    parser.add_argument("--rho-in", type=float, required=True)
    parser.add_argument("--rho-out", type=float, required=True)
    parser.add_argument("--c2c-latency-us", type=float)
    parser.add_argument("--collective-launch-us", type=float)
    parser.add_argument("--lpu-counts", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--tp", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--weight-dtypes", nargs="+", choices=("fp8", "bf16"), default=["fp8", "bf16"]
    )
    parser.add_argument(
        "--kv-dtypes", nargs="+", choices=("fp8", "bf16"), default=["bf16", "fp8"]
    )
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--fit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--step-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    for name, value in (
        ("jobs", args.jobs),
        ("requests", args.requests),
        ("prefill chunk", args.prefill_chunk),
    ):
        integer(value, name, minimum=1)
    integer(args.warmup_requests, "warm-up requests")
    number(args.rho_in, "input tokenizer ratio", positive=True)
    number(args.rho_out, "output tokenizer ratio", positive=True)
    model = HardwareContext.load(ROOT / "data").models.resolve(args.model)
    candidates = []
    for count in args.lpu_counts:
        integer(count, "LPU count", minimum=1)
        for tp in args.tp:
            integer(tp, "TP", minimum=1)
            if (
                count % tp
                or count // tp > model.num_layers
                or model.num_attention_heads % tp
            ):
                continue
            pp = count // tp
            for weights in args.weight_dtypes:
                for kv in args.kv_dtypes:
                    name = f"lpu{count}-tp{tp}-pp{pp}-w{weights}-kv{kv}"
                    candidates.append(
                        dict(
                            name=name,
                            model=args.model,
                            weight_dtype=weights,
                            kv_dtype=kv,
                            layout=asdict(
                                Layout(
                                    a_device="groq_lp30",
                                    f_device="groq_lp30",
                                    a_tp=tp,
                                    a_pp=pp,
                                    f_ep=tp,
                                    f_pp=pp,
                                )
                            ),
                            rho_in=args.rho_in,
                            rho_out=args.rho_out,
                            c2c_latency_us=args.c2c_latency_us,
                            collective_launch_us=args.collective_launch_us,
                            requests=args.requests,
                            warmup=args.warmup_requests,
                            prefill_chunk=args.prefill_chunk,
                            fit=args.fit,
                            step_cache=args.step_cache,
                            lengths=TARGETS,
                            output=str(args.output),
                            resume=args.resume,
                        )
                    )
    args.output.mkdir(parents=True, exist_ok=True)
    revision, dirty = repository_state()
    completed = {}
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(evaluate_candidate, candidate): i
            for i, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            print(
                f"AA done {candidates[index]['name']}: {completed[index]['status']}",
                flush=True,
            )
            write_json(
                args.output / "aa-summary.json",
                dict(
                    git_revision=revision,
                    git_worktree_dirty=dirty,
                    source_targets=TARGETS,
                    tokenizer_assumptions=dict(
                        rho_in=args.rho_in, rho_out=args.rho_out
                    ),
                    results=[completed[i] for i in sorted(completed)],
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
