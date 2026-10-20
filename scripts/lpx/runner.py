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
from scripts.lpx.hardware_overrides import reserve_memory

ROOT = Path(__file__).resolve().parents[2]


def evaluate(job):
    target, workload, case, clients, settings = job
    settings = dict(settings)
    output = settings.pop("checkpoint_output", None)
    resume = settings.pop("resume", True)
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
    return runner.run(parameters, execute).result


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
