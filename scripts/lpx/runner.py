"""Run independent experiment points sequentially or in isolated processes."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from dataclasses import asdict

from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError
from TokenSim.config.validation import integer
from TokenSim.experiments import ExperimentRunner
from scripts.lpx.native_simulation import point
from scripts.lpx.hardware_overrides import reserve_memory, communication_hypothesis

ROOT = Path(__file__).resolve().parents[2]


def evaluate(job):
    target, workload, case, clients, settings = job
    settings = dict(settings)
    output = settings.pop("checkpoint_output", None)
    resume = settings.pop("resume", True)
    result_format = settings.pop("result_format", "full")
    if result_format not in {"full", "reference"}:
        raise ConfigurationError("result format must be full or reference")
    parameters = dict(
        target=target.to_dict(),
        workload=asdict(workload),
        case=asdict(case),
        clients=clients,
        settings={
            key: value.to_dict() if key == "draft_spec" and value is not None else value
            for key, value in settings.items()
        },
    )

    def execute(directory):
        options = dict(settings)
        try:
            backend = options.pop("execution_backend", "native")
            samples = options.pop("replay_samples", 4)
            prefill_samples = options.pop("replay_prefill_samples", 1)
            cache = options.pop(
                "replay_cache", str(ROOT / "outputs/lpx-replay-profile-cache")
            )
            if backend == "sampled-replay":
                from scripts.lpx.replay_driver import ReplayConfig

                options["execution_factory"] = ReplayConfig(
                    workload, samples, prefill_samples, cache
                )
            elif backend != "native":
                raise ConfigurationError("unknown LPX execution backend")
            hardware = reserve_memory(
                HardwareContext.load(ROOT / "data"),
                options.pop("memory_reserve_overrides", {}),
            )
            hardware = communication_hypothesis(
                hardware, **options.pop("communication_overrides", {})
            )
            return point(hardware, target, workload, case, clients, **options)
        except ConfigurationError as exc:
            return dict(
                case=case.name,
                mode=case.mode,
                batch=clients,
                status="infeasible",
                reason=str(exc),
            )

    if output is None:
        if result_format == "reference":
            raise ConfigurationError("result references require a checkpoint directory")
        return execute(None)
    runner = ExperimentRunner(
        output,
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx")
        + (
            (ROOT / "scripts/lpx/replay_engine.cpp",)
            if settings.get("execution_backend") == "sampled-replay"
            else ()
        ),
        resume=resume,
    )
    receipt = runner.run(parameters, execute)
    if result_format == "full":
        return receipt.result
    from scripts.lpx.export import _point_evidence
    from scripts.lpx.validation import validate_point, power_snapshot
    from scripts.lpx.replay_validation import validate_replay_profiles

    hardware = HardwareContext.load(ROOT / "data")
    row = receipt.result
    if row.get("native_executor") == "SampledNativeReplay" and row["status"] == "ok":
        validate_replay_profiles(row)
    validate_point(
        row,
        case,
        workload,
        a_replicas=settings.get("a_dp", 1),
        watts=power_snapshot(hardware, settings.get("power_overrides", {})),
        facility_watts=settings.get("facility_watts", 0),
        model=target.to_dict(),
        executor=(
            "SampledNativeReplay"
            if settings.get("execution_backend") == "sampled-replay"
            else "ModelAFExecutor"
        ),
    )
    compact = _point_evidence(row)
    if row["status"] == "ok":
        compact.pop("turns", None)
        compact["request_statistics"] = {
            key: value
            for key, value in row["request_statistics"].items()
            if key != "requests"
        }
    return dict(
        compact,
        result_file=str(receipt.directory / "result.json"),
        manifest_file=str(receipt.directory / "manifest.json"),
        checkpoint_key=receipt.key,
        reused=receipt.reused,
        point_audit="passed",
    )


def evaluate_jobs(jobs, *, processes=1):
    integer(processes, "experiment processes", minimum=1)
    if processes == 1:
        for index, job in enumerate(jobs):
            yield index, evaluate(job)
        return
    # SimPy clocks, caches and physical resources are independent per point.
    with ProcessPoolExecutor(
        max_workers=processes, mp_context=get_context("spawn")
    ) as pool:
        futures = {pool.submit(evaluate, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            yield futures.pop(future), future.result()
