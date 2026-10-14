"""Af execution."""

from __future__ import annotations

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.llm.llm_request import Request
from tests.af.support.hardware import af_test_hardware


def _requests(env, count, *, phase="decode", prefill_lengths=None):
    lengths = prefill_lengths or [8] * count
    requests = [
        Request(id=index, prefill_len=lengths[index], decode_len=3, block_size=4)
        for index in range(count)
    ]
    for request in requests:
        request.arrive(env)
        if phase == "decode":
            request.generation_idx = 1
    return requests


def af_test_resources(env, cluster, workers=None):
    """Test providers use the same explicit placement as production providers."""
    from TokenSim.af.communication.resources import AFResourceRegistry
    from TokenSim.hardware.placement import build_topology_placement

    workers = cluster.workers() if workers is None else workers
    hardware = af_test_hardware(
        devices=tuple(sorted({w.hardware for w in workers})),
        networks=tuple(sorted({w.nettype for w in workers})),
    )
    placement = (
        build_topology_placement(cluster, workers, hardware)
        if cluster.af.transfer_topology_mode == "physical"
        else None
    )
    return AFResourceRegistry(env, cluster, workers, placement)


def _executor(
    env,
    cluster,
    compute,
    transfer,
    *,
    layers=1,
):
    return AFExecutor(
        env=env,
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        compute_provider=compute,
        transfer_provider=transfer,
        model_id="test-model",
        dtype="fp16",
        total_layers=layers,
        hidden_size=16,
        runtime_plan=execution_plan(cluster, layers=layers, dtype="fp16"),
        resources=af_test_resources(env, cluster, cluster.workers()),
    )


def execution_plan(cluster, *, layers=1, moe=None, dtype="bf16"):
    from TokenSim.af.runtime.compiler import compile_af_runtime_plan

    hardware = af_test_hardware(layers=layers, capacity_gib=0.1, moe=moe)
    return compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        hardware=hardware,
        model=hardware.model("TestModel"),
        block_size=4,
        dtype=dtype,
    )
