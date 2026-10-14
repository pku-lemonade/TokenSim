"""Run independent experiment points sequentially or in isolated processes."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError
from TokenSim.config.validation import integer
from scripts.lpx.native_simulation import point

ROOT = Path(__file__).resolve().parents[2]


def evaluate(job):
    target, workload, case, clients, settings = job
    try:
        return point(
            HardwareContext.load(ROOT / "data"),
            target,
            workload,
            case,
            clients,
            **settings,
        )
    except ConfigurationError as exc:
        return dict(
            case=case.name,
            mode=case.mode,
            batch=clients,
            status="infeasible",
            reason=str(exc),
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
            yield futures[future], future.result()
