"""Shared deterministic hardware, providers and serving fixtures."""

from __future__ import annotations

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.config import ClusterConfig, KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkClass
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request
from TokenSim.workload.source import LLMSource
from tests.af.support.config import _cluster_dict, _with_prefill_pool
from tests.support.hardware import make_device, make_links, make_model, make_topology


AF_TEST_DEVICES = (
    "a-device",
    "f-device",
    "attention-device",
    "ffn-device",
    "prefill-device",
    "decode-device",
    "device",
)


AF_TEST_NETWORKS = (
    "profile",
    "af-network-profile",
    "test",
    "test-network",
    "af-net",
    "net1",
)


def af_test_hardware(
    *,
    layers=1,
    capacity_gib=0.01,
    moe=None,
    hidden_size=16,
    intermediate_size=64,
    num_attention_heads=8,
    devices=AF_TEST_DEVICES,
    networks=AF_TEST_NETWORKS,
):
    """In-memory catalog: tiny devices, the ``TestModel`` architecture and AF links."""
    device_specs = [make_device(name, memory_gib=capacity_gib) for name in devices]
    links = list(make_links()) + [
        LinkClass(
            name,
            f"test link {name}",
            100.0e9,
            1.0,
            source_id="test-fixture",
            grade="D",
            half_bandwidth_bytes=1 << 16,
        )
        for name in networks
        if name not in {"nvlink-test", "ethernet-test"}
    ]
    model = make_model(
        "TestModel",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=layers,
        num_attention_heads=num_attention_heads,
        # The legacy roofline stub had no vocabulary entry and the runtime plan
        # assumed 50000; the hand-calculated component bytes rely on it.
        vocab_size=50000,
        moe=moe,
    )
    return HardwareContext.in_memory(device_specs, [model], links, [make_topology()])


def cluster_test_hardware(cluster, **overrides):
    """Catalog whose device and link names come from ``cluster`` itself."""
    devices = tuple(dict.fromkeys(group.hardware for group in cluster.worker_groups))
    networks = tuple(dict.fromkeys(cluster.networks.values()))
    return af_test_hardware(devices=devices, networks=networks, **overrides)


class FixedComputeProvider:
    def __init__(
        self,
        a: float,
        f: float,
        *,
        router: float | None = None,
        expert: float | None = None,
    ):
        router = f / 3 if router is None else router
        expert = f - router if expert is None else expert
        self.durations = {
            "a": a,
            "router": router,
            "expert": expert,
            "attention": a,
            "dense_ffn": f,
            "routed_expert": expert,
            "shared_expert": expert,
            "weighted_combine": 0.0,
            "residual": 0.0,
            "layout": 0.0,
            "encode": 0.0,
            "decode": 0.0,
        }
        self.queries = []

    def estimate(self, query):
        self.queries.append(query)
        return AFServiceEstimate(
            service_duration=self.durations[query.stage],
            source_id="test-stub:compute",
            details={"stage": query.stage},
        )


class FixedTransferProvider:
    def __init__(
        self,
        a_to_f: float,
        f_to_a: float,
        dispatch: float = 0.0,
        combine: float = 0.0,
        collective: float = 0.0,
        timing_scope: str | None = None,
    ):
        self.timing_scope = timing_scope
        self.durations = {"a_to_f": a_to_f, "f_to_a": f_to_a}
        self.operation_durations = {
            "ep_dispatch": dispatch,
            "ep_combine": combine,
            "tp_collective": collective,
            "all_reduce": collective,
            "all_gather": collective,
            "reduce_scatter": collective,
        }
        self.queries = []

    def estimate(self, query):
        self.queries.append(query)
        duration = self.operation_durations.get(query.operation)
        if duration is None:
            duration = self.durations[query.direction]
        return AFServiceEstimate(
            service_duration=duration,
            source_id="test-stub:transfer",
            payload_bytes=query.payload_bytes,
            details={
                "direction": query.direction,
                "operation": query.operation,
                **({"timing_scope": self.timing_scope} if self.timing_scope else {}),
            },
        )


class InvalidComputeProvider(FixedComputeProvider):
    def __init__(self, invalid_duration):
        super().__init__(invalid_duration, 0.0)


class UnsupportedComputeProvider:
    def estimate(self, query):
        raise LookupError(f"no sample for {query.device_type}:{query.stage}")


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


class _CountingRequest(Request):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class _DTypeCheckingComputeProvider(FixedComputeProvider):
    def __init__(self):
        super().__init__(0.0002, 0.0003)
        self.validated_contracts = []

    def validate_effective_dtypes(self, model_id, effective_dtypes):
        self.validated_contracts.append((model_id, effective_dtypes))


def _typed_cluster(*, trace_enabled=True):
    values = _cluster_dict(trace_enabled=trace_enabled)
    values["af"].update(
        {
            "weight_dtype": "fp8",
            "activation_dtype": "bf16",
            "kv_cache_dtype": "fp32",
        }
    )
    values["af"]["links"][0].update(
        {
            "a_to_f_dtype": "fp8",
            "f_to_a_dtype": "fp16",
        }
    )
    return _with_prefill_pool(ClusterConfig(**values))


def _disabled_cluster():
    return ClusterConfig(
        num_workers=1,
        networks={"local": "profile"},
        worker_groups=[
            {
                "name": "full-pool",
                "capabilities": ["prefill", "decode_full"],
                "hardware": "a-device",
                "num_workers": 1,
                "network": "local",
            }
        ],
    )


def _build_engine(cluster, compute=None, transfer=None):
    cluster = _with_prefill_pool(cluster)
    env = simpy.Environment()
    hardware = af_test_hardware(layers=1, capacity_gib=0.1)
    engine = LLMEngine(
        env=env,
        kv_transfer_config=KVTransferConfig.default(),
        model_spec=hardware.model("TestModel"),
        cluster_config=cluster,
        hardware=hardware,
        af_compute_provider=compute,
        af_transfer_provider=transfer,
        config=SimulationConfig(
            block_size=4,
            batching="paged-attn",
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_num_seqs=8,
        ),
    )
    return env, engine, hardware


def _run_request(env, engine, *, request_id=1, decode_len=3):
    request = _CountingRequest(
        id=request_id,
        prefill_len=4,
        decode_len=decode_len,
        block_size=4,
    )
    env.process(
        LLMSource(
            env=env,
            engine=engine,
            requests=[request],
            qps=1,
            distribution="burst",
        )
    )

    def stop_when_done():
        while env.now < 1.0:
            if request.is_done:
                engine.send_task(engine, Task.STOP)
                yield env.timeout(1e-9)
                return
            yield env.timeout(1e-6)
        raise AssertionError("engine request did not finish")

    monitor = env.process(stop_when_done())
    env.run(until=monitor)
    return request
