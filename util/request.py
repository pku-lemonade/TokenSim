import simpy
import argparse

from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.workload import load_workload


def get_requests(
    args: argparse.Namespace,
    workload_config: WorkloadConfig,
    block_size: int,
    tqdm_submit_func=None,
    *,
    num_experts: int = 0,
) -> list[Request]:
    user_requests = load_workload(
        args=args, workload_config=workload_config, num_experts=num_experts
    )
    requests = user_requests.to_requests(
        block_size=block_size,
        tqdm_submit_func=tqdm_submit_func,
    )
    prefill_lens = user_requests.prefill_lens()
    decode_lens = user_requests.decode_lens()
    return requests, prefill_lens, decode_lens


def LLMSource(
    env: simpy.Environment,
    engine: LLMEngine,
    requests: list[Request],
    qps: float,
    distribution: str,
    random_seed: int = 0,
):
    import numpy as np

    rng = np.random.default_rng(random_seed)
    if any(req.arrival_timestamp is not None for req in requests):
        requests = sorted(
            requests,
            key=lambda req: (
                req.arrival_timestamp
                if req.arrival_timestamp is not None
                else float("inf")
            ),
        )

    for req in requests:
        if req.arrival_timestamp is not None and distribution != "burst":
            yield env.timeout(max(0.0, req.arrival_timestamp - env.now))
        req.arrive(env)
        if distribution == "burst":
            engine.add_requests_burst([req])
        else:
            engine.add_requests([req])
            if req.inter_arrival_time is not None:
                yield env.timeout(req.inter_arrival_time)
            elif req.arrival_timestamp is None:
                yield env.timeout(
                    (
                        1.0 / qps
                        if distribution == "uniform"
                        else float(rng.exponential(1.0 / qps))
                    )
                )
