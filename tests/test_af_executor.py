import math
import sys
import types
import unittest
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import simpy

try:
    import TransformerRoofline  # noqa: F401
except (ImportError, OSError):
    fake_roofline = types.ModuleType("TransformerRoofline")
    fake_roofline.TransformerRoofline = object
    sys.modules["TransformerRoofline"] = fake_roofline

from TokenSim.af import AFExecutor, AFServiceEstimate
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.llm_engine import AFComputeWorker, LLMEngine, Task
from TokenSim.llm.llm_request import Request, reset_g_time
from util.request import LLMSource
from util.results import export_result, get_af_stats

import benchmark


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
        self.durations = {"a": a, "router": router, "expert": expert}
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
    ):
        self.durations = {"a_to_f": a_to_f, "f_to_a": f_to_a}
        self.operation_durations = {
            "ep_dispatch": dispatch,
            "ep_combine": combine,
            "tp_collective": collective,
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
            details={"direction": query.direction, "operation": query.operation},
        )


class InvalidComputeProvider(FixedComputeProvider):
    def __init__(self, invalid_duration):
        super().__init__(invalid_duration, 0.0)


class UnsupportedComputeProvider:
    def estimate(self, query):
        raise LookupError(f"no sample for {query.device_type}:{query.stage}")


class LightweightCacheConfig:
    def __init__(self, block_size, hardware, model, roofline, **kwargs):
        self.block_size = block_size
        self.model = model
        model_config = roofline.models[model]
        self.size_per_token = model_config.Dmodel * 2 * 2 * model_config.Nlayer
        self.num_gpu_blocks = 64
        self.num_cpu_blocks = 64


class AFRooflineStub:
    def __init__(self, reject_full_backend=True, layers=1):
        hardware = SimpleNamespace(
            MM_Card_Num=1,
            Capacity=0.01,
            Nvlink="nvlink-test",
        )
        self.hardwares = {
            "a-device": hardware,
            "f-device": hardware,
            "attention-device": hardware,
            "ffn-device": hardware,
        }
        self.models = {
            "TestModel": SimpleNamespace(
                Name="TestModel",
                Nhead=8,
                Dmodel=16,
                Nlayer=layers,
                Multi_Query=False,
                Grouped_Query=False,
                FFN_Hidden=64,
            )
        }
        self.links = {
            "nvlink-test": SimpleNamespace(Latency=0.0, UniBW=100.0),
            "profile": SimpleNamespace(Latency=0.0, UniBW=100.0),
            "af-network-profile": SimpleNamespace(Latency=0.0, UniBW=100.0),
        }
        self.reject_full_backend = reject_full_backend
        self.full_backend_calls = []

    def Compute_Timebreakdown_Iteration(self, *args, **kwargs):
        self.full_backend_calls.append((args, kwargs))
        if self.reject_full_backend:
            raise AssertionError("AF-selected phases must not call the full backend")
        return 0.0005, 0.0


def _psla_config():
    return PSLAConfig(
        name="af-e2e",
        model="TestModel",
        distribution="burst",
        prefill_mean_len=8,
        prefill_range_len=0,
        decode_mean_len=2,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
    )


def _cluster_dict(
    *,
    phases=("prefill", "decode"),
    af_block_size=1,
    attention_workers=1,
    ffn_workers=1,
    duplex="full",
    trace_enabled=True,
):
    link = {
        "name": "af-link",
        "phases": list(phases),
        "attention_group": "a-pool",
        "ffn_group": "f-pool",
        "network": "af-net",
        "resource_group": "fabric",
        "duplex": duplex,
    }
    if duplex == "full":
        link.update({"a_to_f_capacity": 1, "f_to_a_capacity": 1})
    else:
        link["half_duplex_capacity"] = 1
    worker_groups = [
        {
            "name": "a-pool",
            "capabilities": ["af_attention"],
            "hardware": "a-device",
            "num_workers": attention_workers,
            "network": "af-net",
        },
        {
            "name": "f-pool",
            "capabilities": ["af_ffn"],
            "hardware": "f-device",
            "num_workers": ffn_workers,
            "network": "af-net",
        },
    ]
    if "prefill" not in phases:
        worker_groups.insert(
            0,
            {
                "name": "prefill-pool",
                "role": "prefill",
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            },
        )
    if "decode" not in phases:
        worker_groups.append(
            {
                "name": "decode-pool",
                "role": "decode",
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            }
        )
    return {
        "num_workers": sum(group["num_workers"] for group in worker_groups),
        "networks": {"af-net": "profile"},
        "worker_groups": worker_groups,
        "af": {
            "enabled": True,
            "phases": list(phases),
            "af_block_size": af_block_size,
            "trace_enabled": trace_enabled,
            "links": [link],
        },
    }


def _two_link_cluster(*, shared_resource):
    resource_groups = (
        ("shared-fabric", "shared-fabric")
        if shared_resource
        else (
            "prefill-fabric",
            "decode-fabric",
        )
    )
    return ClusterConfig(
        num_workers=3,
        networks={"af-net": "profile"},
        worker_groups=[
            {
                "name": "prefill-a",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            },
            {
                "name": "decode-a",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            },
            {
                "name": "f-pool",
                "capabilities": ["af_ffn"],
                "hardware": "f-device",
                "num_workers": 1,
                "network": "af-net",
            },
        ],
        af={
            "enabled": True,
            "phases": ["prefill", "decode"],
            "af_block_size": 1,
            "trace_enabled": True,
            "links": [
                {
                    "name": "prefill-link",
                    "phases": ["prefill"],
                    "attention_group": "prefill-a",
                    "ffn_group": "f-pool",
                    "network": "af-net",
                    "resource_group": resource_groups[0],
                    "duplex": "full",
                    "a_to_f_capacity": 1,
                    "f_to_a_capacity": 1,
                },
                {
                    "name": "decode-link",
                    "phases": ["decode"],
                    "attention_group": "decode-a",
                    "ffn_group": "f-pool",
                    "network": "af-net",
                    "resource_group": resource_groups[1],
                    "duplex": "full",
                    "a_to_f_capacity": 1,
                    "f_to_a_capacity": 1,
                },
            ],
        },
    )


def _load_cluster(**overrides):
    return ClusterConfig(**_cluster_dict(**overrides))


def _explicit_domain_cluster(*, with_collective=True):
    values = _cluster_dict(attention_workers=2, ffn_workers=2)
    values["af"]["links"][0].update(
        {"attention_domain": "decode-a", "ffn_domain": "decode-f"}
    )
    physical = lambda group, index=0: {
        "group_name": group,
        "worker_index": index,
    }
    values["af"]["execution_domains"] = [
        {
            "name": "decode-a",
            "capability": "af_attention",
            "phases": ["prefill", "decode"],
            "worker_group": "a-pool",
            "parallel": {"tensor_parallel_size": 2},
            "logical_instances": [
                {
                    "name": "a0",
                    "rank_mapping": [
                        {
                            "domain_rank": rank,
                            "tp_rank": rank,
                            "pp_rank": 0,
                            "dp_rank": 0,
                            "physical_worker": physical("a-pool", rank),
                        }
                        for rank in range(2)
                    ],
                }
            ],
            "communication_plan": (
                [
                    {
                        "operation": "all_reduce",
                        "after_stage": "a_compute",
                        "participants": [0, 1],
                    }
                ]
                if with_collective
                else []
            ),
        },
        {
            "name": "decode-f",
            "capability": "af_ffn",
            "phases": ["prefill", "decode"],
            "worker_group": "f-pool",
            "parallel": {"expert_parallel_size": 2},
            "logical_instances": [
                {
                    "name": "f0",
                    "rank_mapping": [
                        {
                            "domain_rank": index,
                            "tp_rank": 0,
                            "pp_rank": 0,
                            "dp_rank": 0,
                            "ep_rank": index,
                            "physical_worker": physical("f-pool", index),
                        }
                        for index in range(2)
                    ],
                }
            ],
        },
    ]
    return ClusterConfig(**values)


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
    )


class AFExecutorEndToEndTest(unittest.TestCase):
    def test_af_disabled_result_keeps_legacy_json_shape(self):
        reset_g_time()
        cluster = ClusterConfig(num_workers=0, networks={})
        request = Request(id=99, prefill_len=8, decode_len=2, block_size=4)
        engine = SimpleNamespace(
            workers=[],
            af_executor=None,
            af_iteration_summaries=[],
            connector_stats=None,
            parallel_config=None,
        )
        with (
            patch("builtins.open", mock_open()),
            patch.object(Path, "mkdir"),
            patch("util.results.json.dump") as dump,
        ):
            export_result(
                args=Namespace(
                    qps=1,
                    batching="dynamic",
                    results_path="ignored",
                    cluster="unused.json",
                ),
                g_time=SimpleNamespace(
                    request_time=[1.0],
                    prefill_time=[1.0],
                    decode_time=[1.0],
                ),
                engine=engine,
                model_config=_psla_config(),
                cluster=cluster,
                request_count=1,
                prefill_lens=[request.prefill_len],
                decode_lens=[request.decode_len],
                requests=[request],
                notdone=[request.id],
                duration=1.0,
            )

        result = dump.call_args.args[0]
        self.assertNotIn("af_enabled", result)
        self.assertNotIn("af_iteration_count", result)
        self.assertNotIn("af_trace", result)

    def test_llm_engine_runs_prefill_and_decode_through_af_executor(self):
        reset_g_time()
        cluster = ClusterConfig.from_file(
            Path(__file__).parents[1]
            / "data"
            / "clusters"
            / "af"
            / "prefill_decode_full_duplex.json"
        )
        env = simpy.Environment()
        compute = FixedComputeProvider(0.002, 0.003)
        transfer = FixedTransferProvider(0.001, 0.001)
        with patch("TokenSim.llm.llm_engine.CacheConfig", LightweightCacheConfig):
            engine = LLMEngine(
                env=env,
                block_size=4,
                batching="dynamic",
                kv_transfer_config=KVTransferConfig.default(),
                psla_config=_psla_config(),
                cluster_config=cluster,
                roofline=AFRooflineStub(),
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_parallem_sum=8,
                af_compute_provider=compute,
                af_transfer_provider=transfer,
            )
        request = Request(id=10, prefill_len=8, decode_len=2, block_size=4)
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
            while env.now < 0.1:
                if request.is_done:
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-9)
                    return env.now
                yield env.timeout(1e-6)
            raise AssertionError("AF Engine request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)

        self.assertTrue(request.is_done)
        self.assertEqual(request.generation_idx, 2)
        self.assertEqual(len(engine.af_iteration_summaries), 2)
        self.assertEqual(
            [summary["phase"] for summary in engine.af_iteration_summaries],
            ["prefill", "decode"],
        )
        self.assertTrue(
            all(
                math.isclose(summary["makespan"], 0.007)
                for summary in engine.af_iteration_summaries
            )
        )
        self.assertIsInstance(engine.workers[4], AFComputeWorker)
        self.assertFalse(hasattr(engine.workers[4], "scheduler"))
        self.assertEqual(len(compute.queries), 6)
        self.assertEqual(len(transfer.queries), 4)
        af_stats = get_af_stats(engine)
        self.assertTrue(af_stats["af_enabled"])
        self.assertEqual(af_stats["af_iteration_count"], 2)
        self.assertEqual(len(af_stats["af_trace"]), 2)

        with (
            patch("builtins.open", mock_open()),
            patch.object(Path, "mkdir"),
            patch("util.results.json.dump") as dump,
        ):
            export_result(
                args=Namespace(
                    qps=1,
                    batching="dynamic",
                    results_path="ignored",
                    cluster="unused.json",
                ),
                g_time=__import__(
                    "TokenSim.llm.llm_request", fromlist=["g_time"]
                ).g_time,
                engine=engine,
                model_config=_psla_config(),
                cluster=cluster,
                request_count=1,
                prefill_lens=[request.prefill_len],
                decode_lens=[request.decode_len],
                requests=[request],
                notdone=[],
                duration=env.now,
            )
        result = dump.call_args.args[0]
        self.assertTrue(result["af_enabled"])
        self.assertEqual(result["af_iteration_count"], 2)
        self.assertEqual(result["af_trace"][0]["events"][0]["stage"], "a")

    def test_multiblock_multilayer_engine_advances_each_request_once_per_iteration(
        self,
    ):
        reset_g_time()
        cluster = _load_cluster(af_block_size=1)
        env = simpy.Environment()
        compute = FixedComputeProvider(0.0, 0.0)
        transfer = FixedTransferProvider(0.0, 0.0)
        with patch("TokenSim.llm.llm_engine.CacheConfig", LightweightCacheConfig):
            engine = LLMEngine(
                env=env,
                block_size=4,
                batching="dynamic",
                kv_transfer_config=KVTransferConfig.default(),
                psla_config=_psla_config(),
                cluster_config=cluster,
                roofline=AFRooflineStub(layers=2),
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_parallem_sum=16,
                af_compute_provider=compute,
                af_transfer_provider=transfer,
            )
        requests = [
            Request(id=index, prefill_len=8, decode_len=2, block_size=4)
            for index in range(2)
        ]
        env.process(
            LLMSource(
                env=env,
                engine=engine,
                requests=requests,
                qps=1,
                distribution="burst",
            )
        )

        def stop_when_done():
            while env.now < 0.1:
                if all(request.is_done for request in requests):
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-9)
                    return
                yield env.timeout(1e-6)
            raise AssertionError("multiblock AF requests did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)

        self.assertEqual(len(engine.af_iteration_summaries), 2)
        self.assertTrue(
            all(
                len(summary["events"]) == 20
                for summary in engine.af_iteration_summaries
            )
        )
        for request in requests:
            self.assertEqual(request.generation_idx, 2)
            self.assertEqual(request.prefill_batch_size, 2)
            self.assertEqual(request.decode_batch_sum, 2)

    def test_prefill_only_and_decode_only_complete_through_engine_and_kv_transfer(self):
        for selected_phase in ("prefill", "decode"):
            with self.subTest(selected_phase=selected_phase):
                reset_g_time()
                cluster = _load_cluster(phases=(selected_phase,))
                env = simpy.Environment()
                compute = FixedComputeProvider(0.0, 0.0)
                transfer = FixedTransferProvider(0.0, 0.0)
                roofline = AFRooflineStub(reject_full_backend=False)
                with patch(
                    "TokenSim.llm.llm_engine.CacheConfig", LightweightCacheConfig
                ):
                    engine = LLMEngine(
                        env=env,
                        block_size=4,
                        batching="paged-attn",
                        kv_transfer_config=KVTransferConfig.default(),
                        psla_config=_psla_config(),
                        cluster_config=cluster,
                        roofline=roofline,
                        prefill_worker_pool_type="round_robin",
                        decode_worker_pool_type="round_robin",
                        max_parallem_sum=8,
                        af_compute_provider=compute,
                        af_transfer_provider=transfer,
                    )
                request = Request(id=20, prefill_len=8, decode_len=2, block_size=4)
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
                    while env.now < 0.1:
                        if request.is_done:
                            engine.send_task(engine, Task.STOP)
                            yield env.timeout(1e-9)
                            return
                        yield env.timeout(1e-6)
                    raise AssertionError("partial AF request did not finish")

                monitor = env.process(stop_when_done())
                env.run(until=monitor)

                self.assertTrue(request.is_done)
                self.assertEqual(
                    [summary["phase"] for summary in engine.af_iteration_summaries],
                    [selected_phase],
                )
                self.assertGreater(len(roofline.full_backend_calls), 0)
                self.assertGreater(
                    engine.connector_stats.aggregate(
                        engine.request_workers[0].connector.stats
                    ).transfer_count,
                    0,
                )

    def test_complete_and_af_compute_share_a_physical_worker_slot(self):
        values = _cluster_dict(phases=("decode",))
        values["worker_groups"][0] = {
            "name": "a-pool",
            "capabilities": ["prefill", "af_attention"],
            "hardware": "a-device",
            "num_workers": 1,
            "network": "af-net",
        }
        values["worker_groups"].pop(1)
        values["num_workers"] = sum(
            group["num_workers"] for group in values["worker_groups"]
        )
        cluster = ClusterConfig(**values)
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.0),
            FixedTransferProvider(0.0, 0.0),
        )
        iteration = executor.submit(0, _requests(env, 1))

        def complete_compute():
            resource = executor.resources.compute_slots[0]
            with resource.request() as request:
                yield request
                yield env.timeout(0.005)

        env.process(complete_compute())
        env.process(executor.run_iteration(iteration))
        env.run()

        a_event = next(event for event in executor.trace.events if event.stage == "a")
        self.assertAlmostEqual(a_event.ready_at, 0.0)
        self.assertAlmostEqual(a_event.start_at, 0.005)
        self.assertAlmostEqual(a_event.queue_duration, 0.005)

    def test_af_executor_does_not_add_kv_allocate_or_release_operations(self):
        def run_case(af_enabled):
            reset_g_time()
            if af_enabled:
                cluster = _load_cluster()
            else:
                cluster = ClusterConfig(
                    num_workers=1,
                    networks={"af-net": "profile"},
                    worker_groups=[
                        {
                            "name": "full-worker",
                            "capabilities": ["prefill", "decode_full"],
                            "hardware": "a-device",
                            "num_workers": 1,
                            "network": "af-net",
                        }
                    ],
                )
            env = simpy.Environment()
            roofline = AFRooflineStub(reject_full_backend=af_enabled)
            with patch("TokenSim.llm.llm_engine.CacheConfig", LightweightCacheConfig):
                engine = LLMEngine(
                    env=env,
                    block_size=4,
                    batching="paged-attn",
                    kv_transfer_config=KVTransferConfig.default(),
                    psla_config=_psla_config(),
                    cluster_config=cluster,
                    roofline=roofline,
                    prefill_worker_pool_type="round_robin",
                    decode_worker_pool_type="round_robin",
                    max_parallem_sum=8,
                    af_compute_provider=(
                        FixedComputeProvider(0.0, 0.0) if af_enabled else None
                    ),
                    af_transfer_provider=(
                        FixedTransferProvider(0.0, 0.0) if af_enabled else None
                    ),
                )
            manager = engine.request_workers[0].scheduler.block_manager
            counts = {"allocate": 0, "release": 0}
            original_allocate = manager.allocate
            original_release = manager.release_request_blocks

            def allocate(request):
                counts["allocate"] += 1
                return original_allocate(request)

            def release(request):
                counts["release"] += 1
                return original_release(request)

            manager.allocate = allocate
            manager.release_request_blocks = release
            request = Request(id=30, prefill_len=8, decode_len=2, block_size=4)
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
                while env.now < 0.1:
                    if request.is_done:
                        engine.send_task(engine, Task.STOP)
                        yield env.timeout(1e-9)
                        return
                    yield env.timeout(1e-6)
                raise AssertionError("KV ownership comparison did not finish")

            monitor = env.process(stop_when_done())
            env.run(until=monitor)
            self.assertTrue(request.is_done)
            return counts

        self.assertEqual(run_case(True), run_case(False))

    def test_repository_af_json_loads_through_real_config_entrypoint(self):
        path = (
            Path(__file__).parents[1]
            / "data"
            / "clusters"
            / "af"
            / "prefill_decode_full_duplex.json"
        )

        cluster = ClusterConfig.from_file(path)

        self.assertTrue(cluster.af.enabled)
        self.assertEqual(cluster.af.phases, ("prefill", "decode"))

    def test_real_config_builds_prefill_and_decode_blocks(self):
        cluster = _load_cluster(af_block_size=2)
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.0, 0.0),
            FixedTransferProvider(0.0, 0.0),
        )

        prefill = _requests(env, 5, phase="prefill", prefill_lengths=[1, 2, 3, 4, 5])
        iteration = executor.submit(0, prefill)

        self.assertEqual([block.request_count for block in iteration.blocks], [2, 2, 1])
        self.assertEqual([block.token_count for block in iteration.blocks], [3, 7, 5])
        self.assertEqual(iteration.request_ids, (0, 1, 2, 3, 4))
        self.assertEqual(iteration.phase, "prefill")

        decode = _requests(env, 5, phase="decode")
        decode_iteration = executor.submit(0, decode)
        self.assertEqual(
            [block.token_count for block in decode_iteration.blocks], [2, 2, 1]
        )

    def test_single_block_oracle_is_seven_milliseconds(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        compute = FixedComputeProvider(0.002, 0.003)
        transfer = FixedTransferProvider(0.001, 0.001)
        executor = _executor(env, cluster, compute, transfer)
        iteration = executor.submit(0, _requests(env, 1))

        env.process(executor.run_iteration(iteration))
        env.run()

        self.assertAlmostEqual(iteration.makespan, 0.007)
        events = executor.trace.events_for(iteration.iteration_id)
        self.assertEqual(
            [event.stage or event.operation for event in events],
            [
                "a",
                "router",
                "af_a_to_f",
                "expert",
                "af_f_to_a",
            ],
        )
        self.assertTrue(all(event.queue_duration == 0 for event in events))
        self.assertTrue(
            all(event.source_id.startswith("test-stub:") for event in events)
        )
        self.assertEqual(events[2].payload_bytes, 32)
        self.assertTrue(
            all(
                math.isclose(event.queue_duration, event.start_at - event.ready_at)
                and math.isclose(event.service_duration, event.end_at - event.start_at)
                for event in events
            )
        )
        self.assertEqual(compute.queries[0].phase, "decode")
        self.assertEqual(compute.queries[0].device_type, "a-device")
        self.assertEqual(compute.queries[1].device_type, "a-device")
        self.assertEqual(compute.queries[2].device_type, "f-device")
        self.assertEqual(transfer.queries[0].network, "af-net")
        self.assertEqual(
            [query.operation for query in transfer.queries],
            ["af_a_to_f", "af_f_to_a"],
        )

    def test_single_block_oracle_merges_dispatch_into_cross_domain_transfer(self):
        cluster = _explicit_domain_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.003, router=0.001, expert=0.002),
            FixedTransferProvider(0.001, 0.001, dispatch=0.001, combine=0.001),
        )
        iteration = executor.submit(0, _requests(env, 1))

        env.process(executor.run_iteration(iteration))
        env.run()

        self.assertAlmostEqual(iteration.makespan, 0.007)
        events = executor.trace.events_for(iteration.iteration_id)
        self.assertEqual(
            [event.stage or event.operation for event in events],
            [
                "a",
                "a",
                "tp_collective",
                "router",
                "af_a_to_f",
                "expert",
                "af_f_to_a",
            ],
        )
        self.assertLess(events[3].end_at, events[-1].end_at)
        self.assertEqual(
            [query.stage for query in executor.compute_provider.queries],
            ["a", "a", "router", "expert"],
        )
        self.assertTrue(
            all(event.source_domain and event.target_domain for event in events)
        )
        communication = [event for event in events if event.event_type == "transfer"]
        self.assertTrue(all(event.physical_endpoints for event in communication))
        self.assertTrue(all(event.resource_ids for event in communication))

    def test_explicit_domain_plan_runs_collective_without_fictitious_capacity(self):
        cluster = _explicit_domain_cluster()
        env = simpy.Environment()
        transfer = FixedTransferProvider(
            0.001,
            0.001,
            dispatch=0.001,
            combine=0.001,
            collective=0.001,
        )
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.003),
            transfer,
        )
        iteration = executor.submit(0, _requests(env, 2))

        env.process(executor.run_iteration(iteration))
        env.run()

        events = executor.trace.events_for(iteration.iteration_id)
        collectives = [event for event in events if event.event_type == "collective"]
        self.assertEqual(len(collectives), 2)
        self.assertTrue(
            all(event.operation == "tp_collective" for event in collectives)
        )
        self.assertTrue(all(event.participant_ranks == (0, 1) for event in collectives))
        a_events = [event for event in events if event.stage == "a"]
        self.assertTrue(
            all(
                len(event.physical_endpoints) == 1
                and event.physical_endpoints[0] in {"a-pool:0", "a-pool:1"}
                for event in a_events
            )
        )
        self.assertTrue(all(len(event.resource_ids) == 1 for event in a_events))
        self.assertEqual({event.participant_ranks for event in a_events}, {(0,), (1,)})
        by_block = {}
        for event in a_events:
            by_block.setdefault(event.block_id, []).append(event)
        self.assertTrue(all(len(events) == 2 for events in by_block.values()))
        self.assertTrue(
            all(
                events[0].start_at == events[1].start_at for events in by_block.values()
            )
        )
        self.assertEqual(
            [query.operation for query in transfer.queries].count("tp_collective"), 2
        )

        no_collective = _explicit_domain_cluster(with_collective=False)
        no_collective_env = simpy.Environment()
        no_collective_transfer = FixedTransferProvider(0.0, 0.0)
        no_collective_executor = _executor(
            no_collective_env,
            no_collective,
            FixedComputeProvider(0.0, 0.0),
            no_collective_transfer,
        )
        no_collective_iteration = no_collective_executor.submit(
            0, _requests(no_collective_env, 1)
        )
        no_collective_env.process(
            no_collective_executor.run_iteration(no_collective_iteration)
        )
        no_collective_env.run()
        self.assertIn(
            "tp_collective",
            [query.operation for query in no_collective_transfer.queries],
        )

    def test_engine_gives_one_scheduler_to_a_logical_instance(self):
        cluster = _explicit_domain_cluster()
        env = simpy.Environment()
        with patch("TokenSim.llm.llm_engine.CacheConfig", LightweightCacheConfig):
            engine = LLMEngine(
                env=env,
                block_size=4,
                batching="dynamic",
                kv_transfer_config=KVTransferConfig.default(),
                psla_config=_psla_config(),
                cluster_config=cluster,
                roofline=AFRooflineStub(),
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_parallem_sum=8,
                af_compute_provider=FixedComputeProvider(0.0, 0.0),
                af_transfer_provider=FixedTransferProvider(0.0, 0.0),
            )

        self.assertEqual([worker.id for worker in engine.request_workers], [0])
        self.assertIsInstance(engine.workers[1], AFComputeWorker)
        self.assertIsInstance(engine.workers[2], AFComputeWorker)

    def test_two_block_oracle_forms_eleven_millisecond_pipeline(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.003),
            FixedTransferProvider(0.001, 0.001),
        )
        iteration = executor.submit(0, _requests(env, 2))

        env.process(executor.run_iteration(iteration))
        env.run()

        self.assertAlmostEqual(iteration.makespan, 0.011)
        events = executor.trace.events_for(iteration.iteration_id)
        block_zero_expert = next(
            event for event in events if event.block_id == 0 and event.stage == "expert"
        )
        block_one_a = next(
            event for event in events if event.block_id == 1 and event.stage == "a"
        )
        self.assertLess(block_one_a.start_at, block_zero_expert.end_at)
        self.assertEqual(iteration.barrier_wait[1], 0.0)
        self.assertGreater(iteration.barrier_wait[0], 0.0)
        summary = executor.trace.summarize(iteration)
        self.assertAlmostEqual(summary["resource_utilization"]["worker:0"], 6 / 11)
        self.assertAlmostEqual(summary["resource_utilization"]["worker:1"], 4 / 11)

    def test_two_a_workers_keep_affinity_and_share_engine_resources(self):
        cluster = _load_cluster(attention_workers=2, ffn_workers=2)
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.0),
            FixedTransferProvider(0.001, 0.0, dispatch=0.001),
        )
        first = executor.submit(0, _requests(env, 1))
        second = executor.submit(1, _requests(env, 1))

        env.process(executor.run_iteration(first))
        env.process(executor.run_iteration(second))
        env.run()

        first_a = next(
            event
            for event in executor.trace.events_for(first.iteration_id)
            if event.stage == "a"
        )
        second_a = next(
            event
            for event in executor.trace.events_for(second.iteration_id)
            if event.stage == "a"
        )
        transfers = [
            event for event in executor.trace.events if event.direction == "a_to_f"
        ]
        self.assertEqual((first_a.worker_id, second_a.worker_id), (0, 1))
        self.assertEqual((first_a.start_at, second_a.start_at), (0, 0))
        self.assertTrue(any(event.queue_duration > 0 for event in transfers))
        routers = [event for event in executor.trace.events if event.stage == "router"]
        self.assertEqual({event.worker_id for event in routers}, {0, 1})
        self.assertNotIn(
            "ep_dispatch", {event.operation for event in executor.trace.events}
        )

    def test_same_resource_group_contends_across_phase_links(self):
        makespans = {}
        for shared in (True, False):
            cluster = _two_link_cluster(shared_resource=shared)
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(0.0, 0.0),
                FixedTransferProvider(0.003, 0.003),
            )
            prefill = executor.submit(0, _requests(env, 1, phase="prefill"))
            decode = executor.submit(1, _requests(env, 1, phase="decode"))
            env.process(executor.run_iteration(prefill))
            env.process(executor.run_iteration(decode))
            env.run()
            makespans[shared] = max(prefill.end_at, decode.end_at)

        self.assertGreater(makespans[True], makespans[False])

    def test_a_and_f_bottlenecks_are_visible_in_queue_time(self):
        queue_totals = {}
        for name, a_duration, f_duration in (
            ("a-slow", 0.004, 0.001),
            ("f-slow", 0.001, 0.004),
        ):
            cluster = _load_cluster()
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(a_duration, f_duration),
                FixedTransferProvider(0.0, 0.0),
            )
            iteration = executor.submit(0, _requests(env, 3))
            env.process(executor.run_iteration(iteration))
            env.run()
            events = executor.trace.events_for(iteration.iteration_id)
            queue_totals[name] = {
                stage: sum(
                    event.queue_duration for event in events if event.stage == stage
                )
                for stage in ("a", "router", "expert")
            }

        self.assertGreater(queue_totals["a-slow"]["a"], 0)
        self.assertGreater(queue_totals["f-slow"]["expert"], 0)

    def test_multiple_layers_wait_for_previous_return(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.003),
            FixedTransferProvider(0.001, 0.001),
            layers=2,
        )
        iteration = executor.submit(0, _requests(env, 1))

        env.process(executor.run_iteration(iteration))
        env.run()

        events = executor.trace.events_for(iteration.iteration_id)
        layer_zero_return = next(
            event
            for event in events
            if event.layer_id == 0
            and event.direction == "f_to_a"
            and event.operation == "af_f_to_a"
        )
        layer_one_a = next(
            event for event in events if event.layer_id == 1 and event.stage == "a"
        )
        self.assertEqual(layer_one_a.ready_at, layer_zero_return.end_at)
        self.assertAlmostEqual(iteration.makespan, 0.014)

    def test_full_duplex_overlaps_opposite_directions_but_half_does_not(self):
        makespans = {}
        traces = {}
        for duplex in ("full", "half"):
            cluster = _load_cluster(duplex=duplex)
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(0.001, 0.0),
                FixedTransferProvider(0.003, 0.003),
            )
            iteration = executor.submit(0, _requests(env, 2))
            env.process(executor.run_iteration(iteration))
            env.run()
            makespans[duplex] = iteration.makespan
            traces[duplex] = executor.trace.events_for(iteration.iteration_id)

        full_a_to_f = next(
            event
            for event in traces["full"]
            if event.block_id == 1
            and event.direction == "a_to_f"
            and event.operation == "af_a_to_f"
        )
        full_f_to_a = next(
            event
            for event in traces["full"]
            if event.block_id == 0
            and event.direction == "f_to_a"
            and event.operation == "af_f_to_a"
        )
        self.assertNotEqual(full_a_to_f.resource_id, full_f_to_a.resource_id)
        half_transfers = [
            event for event in traces["half"] if event.event_type == "transfer"
        ]
        service_intervals = sorted(
            (event.start_at, event.end_at) for event in half_transfers
        )
        self.assertTrue(
            all(
                left[1] <= right[0]
                for left, right in zip(service_intervals, service_intervals[1:])
            )
        )
        self.assertGreaterEqual(makespans["half"], makespans["full"])

    def test_multiple_f_workers_reduce_the_f_bottleneck(self):
        makespans = {}
        f_events = {}
        for ffn_workers in (1, 2):
            cluster = _load_cluster(ffn_workers=ffn_workers)
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(0.0, 0.003),
                FixedTransferProvider(0.0, 0.0),
            )
            iteration = executor.submit(0, _requests(env, 2))
            env.process(executor.run_iteration(iteration))
            env.run()
            makespans[ffn_workers] = iteration.makespan
            f_events[ffn_workers] = [
                event
                for event in executor.trace.events_for(iteration.iteration_id)
                if event.stage in {"router", "expert"}
            ]

        self.assertGreater(makespans[1], makespans[2])
        self.assertEqual(
            {event.worker_id for event in f_events[1] if event.stage == "expert"},
            {1},
        )
        self.assertEqual(
            {event.worker_id for event in f_events[2] if event.stage == "expert"},
            {1, 2},
        )
        self.assertEqual(
            [event.start_at for event in f_events[2] if event.stage == "router"],
            [0.0, 0.001],
        )
        self.assertEqual(
            {event.worker_id for event in f_events[2] if event.stage == "router"},
            {0},
        )

    def test_trace_disabled_does_not_change_timing(self):
        results = {}
        for enabled in (True, False):
            cluster = _load_cluster(trace_enabled=enabled)
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(0.002, 0.003),
                FixedTransferProvider(0.001, 0.001),
            )
            iteration = executor.submit(0, _requests(env, 2))
            env.process(executor.run_iteration(iteration))
            env.run()
            results[enabled] = (iteration.makespan, len(executor.trace.events))

        self.assertEqual(results[True][0], results[False][0])
        self.assertGreater(results[True][1], 0)
        self.assertEqual(results[False][1], 0)

    def test_invalid_provider_values_fail_with_context_and_release_resources(self):
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                cluster = _load_cluster()
                env = simpy.Environment()
                compute = InvalidComputeProvider(invalid)
                executor = _executor(
                    env,
                    cluster,
                    compute,
                    FixedTransferProvider(0.0, 0.0),
                )
                iteration = executor.submit(0, _requests(env, 2))
                env.process(executor.run_iteration(iteration))

                with self.assertRaisesRegex(
                    SimulationStateError, r"block=[01], layer=0, stage=a"
                ):
                    env.run()

                self.assertEqual(executor.active_iterations, {})
                self.assertTrue(
                    all(
                        resource.count == 0
                        for resource in executor.resources.compute_slots.values()
                    )
                )
                self.assertTrue(
                    all(
                        resource.count == 0
                        for resource in executor.resources.communication_resources.values()
                    )
                )
                executor.compute_provider = FixedComputeProvider(0.0, 0.0)
                recovered = executor.submit(0, _requests(env, 1))
                env.process(executor.run_iteration(recovered))
                env.run()
                self.assertEqual(recovered.makespan, 0.0)

    def test_unsupported_provider_fails_without_full_backend_fallback(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            UnsupportedComputeProvider(),
            FixedTransferProvider(0.0, 0.0),
        )
        iteration = executor.submit(0, _requests(env, 1))
        env.process(executor.run_iteration(iteration))

        with self.assertRaisesRegex(
            SimulationStateError,
            r"unsupported.*block=0, layer=0, stage=a.*no sample",
        ):
            env.run()

        self.assertEqual(executor.active_iterations, {})
        self.assertEqual(executor.trace.events, [])

    def test_event_queue_exhaustion_reports_active_af_stage(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.001, 0.0),
            FixedTransferProvider(0.0, 0.0),
        )
        request = _requests(env, 1)[0]
        iteration = executor.submit(0, [request])

        def hold_a_slot_forever():
            resource = executor.resources.compute_slots[0]
            with resource.request() as resource_request:
                yield resource_request
                yield env.event()

        env.process(hold_a_slot_forever())
        env.process(executor.run_iteration(iteration))
        env.run()
        self.assertEqual(len(executor.active_iterations), 1)

        args = Namespace(
            cluster="af.json",
            qps=1,
            results_path="diagnostics",
        )
        engine = SimpleNamespace(af_executor=executor, workers=[])
        with (
            patch.object(benchmark.Path, "mkdir"),
            patch.object(benchmark.Path, "write_text") as write_text,
            patch.object(benchmark.os, "replace"),
            patch("builtins.open", mock_open()),
            self.assertRaisesRegex(SimulationStateError, "unfinished requests"),
        ):
            benchmark.check_results(
                args=args,
                requests=[request],
                engine=engine,
                model_config=MagicMock(),
                cluster=cluster,
                duration=env.now,
                request_count=1,
                prefill_lens=[request.prefill_len],
                decode_lens=[request.decode_len],
                simulator_wall_time=0.0,
            )
        failure = json.loads(write_text.call_args.args[0])

        snapshots = failure["unfinished_af_iterations"]
        self.assertEqual(snapshots[0]["iteration_id"], iteration.iteration_id)
        self.assertEqual(snapshots[0]["completed_block_ids"], [])
        self.assertEqual(snapshots[0]["active_stages"][0]["stage"], "a")

    def test_missing_runtime_dependencies_fail_before_events(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        with self.assertRaisesRegex(ConfigurationError, "compute latency provider"):
            _executor(
                env,
                cluster,
                None,
                FixedTransferProvider(0.0, 0.0),
            )

    def test_decode_only_rejects_prefill_submission(self):
        cluster = _load_cluster(phases=("decode",))
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.0, 0.0),
            FixedTransferProvider(0.0, 0.0),
        )
        source_worker_id = cluster.worker_groups[0].num_workers

        with self.assertRaisesRegex(ConfigurationError, "not selected"):
            executor.submit(source_worker_id, _requests(env, 1, phase="prefill"))


if __name__ == "__main__":
    unittest.main()
