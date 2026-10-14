"""Serving integration, phase selection, mixed batches, budgets and CLI export."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import simpy

from TokenSim.cli import build_parser, main
from TokenSim.comm.collectives import CollectiveModel
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import (
    ConfigurationError,
    SimulationStateError,
    WorkloadValidationError,
)
from TokenSim.hardware.links import LinkCatalog
from TokenSim.hardware.topology import TopologyLevel, TopologySpec
from TokenSim.llm.engine.af_worker import run_worker_compute
from TokenSim.llm.llm_engine import LLMEngine, LLMWorker
from TokenSim.llm.llm_request import Request
from TokenSim.moe import ExpertRouting, normalize_expert_histogram
from TokenSim.reporting import get_af_stats
from tests.af.support.config import _cluster, _cluster_dict
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    _DTypeCheckingComputeProvider,
    _build_engine,
    _disabled_cluster,
    _run_request,
    _typed_cluster,
    af_test_hardware,
)
from tests.af.support.model import _moe
from tests.support.hardware import make_links, moe_test_model


class AFEngineIntegrationTest(unittest.TestCase):
    def test_decode_only_connects_runtime_plan_dtypes_and_providers(self):
        compute = _DTypeCheckingComputeProvider()
        transfer = FixedTransferProvider(0.0001, 0.0001)
        env, engine, roofline = _build_engine(_typed_cluster(), compute, transfer)

        plan = engine.af_runtime_plan
        self.assertIsNotNone(plan)
        self.assertIs(engine.af_executor.runtime_plan, plan)
        self.assertEqual(
            {
                "weight_dtype": plan.effective_dtypes.weight_dtype,
                "activation_dtype": plan.effective_dtypes.activation_dtype,
                "kv_cache_dtype": plan.effective_dtypes.kv_cache_dtype,
                "a_to_f_dtype": plan.effective_dtypes.a_to_f_dtype,
                "f_to_a_dtype": plan.effective_dtypes.f_to_a_dtype,
            },
            {
                "weight_dtype": "fp8",
                "activation_dtype": "bf16",
                "kv_cache_dtype": "fp32",
                "a_to_f_dtype": "fp8",
                "f_to_a_dtype": "fp16",
            },
        )
        self.assertEqual(
            [(layer.layer_id, layer.layer_type) for layer in plan.layers],
            [(0, "dense")],
        )
        self.assertEqual(len(compute.validated_contracts), 1)
        self.assertEqual(compute.validated_contracts[0][0], "TestModel")

        decode_manager = engine.decode_workers.workers[0].scheduler.block_manager
        with patch.object(
            decode_manager,
            "commit_input_cache",
            wraps=decode_manager.commit_input_cache,
        ) as commit_input_cache:
            request = _run_request(env, engine)

        self.assertTrue(request.is_done)
        self.assertEqual(request.generation_idx, 3)
        self.assertEqual(request.step_calls, 3)
        self.assertEqual(commit_input_cache.call_count, 2)
        self.assertEqual(
            [item["phase"] for item in engine.af_iteration_summaries],
            ["decode", "decode"],
        )
        self.assertEqual({query.phase for query in compute.queries}, {"decode"})
        self.assertEqual({query.phase for query in transfer.queries}, {"decode"})
        self.assertTrue({query.stage for query in compute.queries})
        for query in compute.queries:
            self.assertEqual(query.weight_dtype, "fp8")
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.kv_cache_dtype, "fp32")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp16")
        for query in transfer.queries:
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp16")
            if query.direction == "a_to_f":
                self.assertEqual(query.dtype, "fp8")
            elif query.direction == "f_to_a":
                self.assertEqual(query.dtype, "fp16")
        self.assertEqual(
            get_af_stats(engine)["af_phase_fidelity"],
            {
                "prefill": "prefill_aggregate",
                "decode": "decode_af_rank_local",
            },
        )

    def test_af_prefill_uses_compatibility_path_and_full_decode(self):
        compute = _DTypeCheckingComputeProvider()
        transfer = FixedTransferProvider(0.0001, 0.0001)
        env, engine, roofline = _build_engine(
            _typed_cluster(phases=("prefill",)), compute, transfer
        )

        request = _run_request(env, engine, decode_len=2)

        self.assertTrue(request.is_done)
        self.assertEqual(request.step_calls, 2)
        self.assertEqual(
            [item["phase"] for item in engine.af_iteration_summaries],
            ["prefill"],
        )
        self.assertEqual({query.phase for query in compute.queries}, {"prefill"})
        self.assertEqual({query.phase for query in transfer.queries}, {"prefill"})
        self.assertEqual(
            get_af_stats(engine)["af_phase_fidelity"],
            {
                "prefill": "prefill_af_rank_local",
                "decode": "decode_aggregate",
            },
        )

    def test_af_disabled_keeps_complete_backend_and_creates_no_af_state(self):
        env, engine, roofline = _build_engine(_disabled_cluster())

        request = _run_request(env, engine, decode_len=2)

        self.assertTrue(request.is_done)
        self.assertEqual(request.step_calls, 2)
        self.assertIsNone(engine.af_runtime_plan)
        self.assertIsNone(engine.af_executor)
        self.assertEqual(
            get_af_stats(engine),
            {
                "af_enabled": False,
                "af_iteration_count": 0,
                "af_trace": None,
                "af_runtime_state": None,
                "af_phase_fidelity": None,
                "af_latency_providers": None,
                "af_operator_stats": None,
            },
        )


class AFServingCorrectnessTest(unittest.TestCase):
    def make_engine(self, *, utilization=1.0, capacity=None, total=None, replicas=1):
        hardware = af_test_hardware(capacity_gib=0.1)
        cluster = _cluster(a_tp=1, a_dp=replicas, f_tp=1, f_dp=1, f_ep=1)
        cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
        cluster.kv_cache_capacity_tokens_total = total
        return LLMEngine(
            env=simpy.Environment(),
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=4, batching="paged-attn", gpu_memory_utilization=utilization
            ),
            af_compute_provider=FixedComputeProvider(0.001, 0.001),
            af_transfer_provider=FixedTransferProvider(0.001, 0.001),
        )

    def test_memory_utilization_limits_every_physical_budget(self):
        full = self.make_engine()
        half = self.make_engine(utilization=0.5)
        self.assertLess(
            half.request_workers[0].cache_config.num_gpu_blocks,
            full.request_workers[0].cache_config.num_gpu_blocks // 2,
        )
        for original, limited in zip(
            full.af_runtime_plan.physical_states, half.af_runtime_plan.physical_states
        ):
            self.assertEqual(limited.capacity_bytes, original.capacity_bytes // 2)

    def test_explicit_kv_limit_caps_all_attention_shards(self):
        engine = self.make_engine(capacity=32)
        manager = engine.request_workers[0].scheduler.block_manager
        self.assertEqual(manager.num_total_gpu_blocks, 8)
        for rank in manager.rank_keys:
            self.assertEqual(manager.rank_status(rank)["gpu_capacity"], 8)
        with self.assertRaisesRegex(ConfigurationError, "cannot fit"):
            engine.validate_request_capacity([Request(1, 64, 2, 4)])

    def test_kv_limit_cannot_manufacture_physical_memory(self):
        full = self.make_engine()
        capped = self.make_engine(capacity=10**12)
        self.assertEqual(
            full.request_workers[0].cache_config.num_gpu_blocks,
            capped.request_workers[0].cache_config.num_gpu_blocks,
        )

    def test_lockstep_is_rejected_before_starting_workers(self):
        for ranks in (1, 2):
            with self.subTest(ranks=ranks):
                env = simpy.Environment()
                hardware = af_test_hardware(capacity_gib=0.1)
                with self.assertRaisesRegex(ConfigurationError, "dp_lockstep.*AF"):
                    LLMEngine(
                        env=env,
                        kv_transfer_config=KVTransferConfig.default(),
                        model_spec=hardware.model("TestModel"),
                        cluster_config=_cluster(
                            a_tp=1, a_dp=ranks, f_tp=1, f_dp=1, f_ep=1
                        ),
                        hardware=hardware,
                        parallel_config=ParallelConfig(data_parallel_size=ranks),
                        config=SimulationConfig(dp_lockstep=True),
                    )
                self.assertEqual(len(env._queue), 0)

    def test_total_limit_is_divided_by_attention_domain_replicas(self):
        engine = self.make_engine(total=64, replicas=2)
        self.assertEqual(engine.parallel_config.data_parallel_size, 1)
        self.assertEqual(len(engine.request_workers), 2)
        self.assertEqual(
            [w.cache_config.num_gpu_blocks for w in engine.request_workers], [8, 8]
        )
        with self.assertRaisesRegex(ConfigurationError, "one KV block per A replica"):
            self.make_engine(total=7, replicas=2)


class MixedServingTest(unittest.TestCase):
    def engine(self, *, capacity=None, compute=None, moe=False):
        cluster = ClusterConfig(**_cluster_dict(af_block_size=8))
        cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
        hardware = af_test_hardware(capacity_gib=0.1, moe=_moe() if moe else None)
        env = simpy.Environment()
        compute = compute or FixedComputeProvider(0.001, 0.002)
        engine = LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=4, batching="paged-attn", max_num_seqs=2
            ),
            af_compute_provider=compute,
            af_transfer_provider=FixedTransferProvider(0.001, 0.001),
        )
        return env, engine, compute

    def test_staggered_moe_requests_never_use_the_complete_backend(self):
        env, engine, _ = self.engine(moe=True)
        requests = [Request(1, 8, 4, 4), Request(2, 8, 3, 4)]
        batches = []
        original = LLMWorker.step

        def step(worker, running, latency):
            batches.append(tuple(req.is_prefill for req in running))
            return original(worker, running, latency)

        def source():
            requests[0].arrive(env)
            engine.add_requests([requests[0]])
            yield env.timeout(0.001)
            requests[1].arrive(env)
            engine.add_requests([requests[1]])

        with (
            patch.object(
                LLMWorker,
                "dynamic_batch",
                side_effect=AssertionError("complete backend called"),
            ),
            patch.object(LLMWorker, "step", step),
        ):
            env.process(source())
            env.run()
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        self.assertTrue(any(len(set(batch)) == 2 for batch in batches))
        self.assertEqual(len(engine.timing_recorder.timing.time), 2)

    def test_memory_pressure_recomputes_through_the_af_graph(self):
        env, engine, compute = self.engine(capacity=16)
        requests = [Request(i, 8, 8, 4) for i in (1, 2)]
        with patch.object(
            LLMWorker,
            "dynamic_batch",
            side_effect=AssertionError("complete backend called"),
        ):
            for req in requests:
                req.arrive(env)
                engine.add_requests_burst([req])
            env.run()
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        self.assertGreater(sum(req.recomputed_tokens_total for req in requests), 0)
        self.assertTrue(
            any(
                q.phase == "prefill" and q.context_lengths[0] > 8
                for q in compute.queries
            )
        )
        self.assertEqual(engine.af_executor.active_iterations, {})

    def test_failed_group_cancels_other_inflight_groups_and_releases_resources(self):
        class FailPrefill(FixedComputeProvider):
            def estimate(self, query):
                if query.phase == "prefill":
                    raise ValueError("prefill failed")
                return super().estimate(query)

        env, engine, _ = self.engine(compute=FailPrefill(0.001, 0.002))
        decode, prefill = Request(1, 8, 4, 4), Request(2, 8, 4, 4)
        decode.generation_idx = 1
        process = env.process(
            run_worker_compute(engine.request_workers[0], [decode, prefill])
        )
        with self.assertRaisesRegex(SimulationStateError, "prefill failed"):
            env.run(until=process)
        self.assertEqual(engine.af_executor.active_iterations, {})
        for resource in engine.af_resources.compute_slots.values():
            self.assertEqual(resource.count, 0)
            self.assertFalse(resource.queue)


class AFMergeCompatibilityTest(unittest.TestCase):
    def test_point_to_point_oversubscription_never_improves_latency(self):
        self.links = LinkCatalog(make_links())

        topology = TopologySpec(
            "three-level",
            (
                TopologyLevel("node", 4, "nvlink-test", links_per_endpoint=2),
                TopologyLevel("rack", 2, "ethernet-test"),
                TopologyLevel("cluster", None, "ethernet-test"),
            ),
        )
        for endpoints, level_index in (((0, 1), 0), ((0, 4), 1), ((0, 8), 2)):
            for payload in (0, 1, 1 << 20):
                with self.subTest(endpoints=endpoints, payload=payload):
                    layout = topology.group_layout(endpoints)
                    self.assertEqual(layout.lowest_common_level, level_index)
                    previous = -1
                    for oversubscription in (1, 2, 4):
                        levels = list(topology.levels)
                        levels[level_index] = replace(
                            levels[level_index], oversubscription=oversubscription
                        )
                        model = CollectiveModel(
                            replace(topology, levels=tuple(levels)), self.links
                        )
                        estimate = model.point_to_point(payload, layout)
                        self.assertGreaterEqual(estimate.latency_us, previous)
                        previous = estimate.latency_us
                        if payload:
                            self.assertEqual(
                                estimate.levels[0].level, levels[level_index].name
                            )

    def test_routing_validates_each_phase_and_does_not_reuse_prefill_cache(self):
        routing = ExpertRouting(moe_test_model().moe, seed=1)
        req = Request(
            0,
            prefill_len=4,
            decode_len=2,
            block_size=16,
            prefill_expert_histogram={0: 8},
            decode_expert_histogram={1: 2},
        )

        self.assertEqual(routing.histogram_for_step([req]), {0: 8})
        req.generation_idx = 1
        self.assertEqual(routing.histogram_for_step([req]), {1: 2})

        bad = Request(
            1,
            prefill_len=4,
            decode_len=1,
            block_size=16,
            prefill_expert_histogram={0: 7},
        )
        with self.assertRaisesRegex(WorkloadValidationError, "must equal 8"):
            routing.histogram_for_step([bad])

    def test_histogram_rejects_an_incorrect_route_count(self):
        with self.assertRaisesRegex(WorkloadValidationError, "must equal 8"):
            normalize_expert_histogram(
                {0: 3, 1: 4}, num_experts=4, request_id=7, expected_route_count=8
            )

    def test_current_cli_runs_af_and_exports_runtime_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cluster_path = path / "cluster.json"
            cluster_path.write_text(json.dumps(_cluster_dict()))
            args = build_parser().parse_args(
                [
                    "--cluster",
                    str(cluster_path),
                    "--model",
                    "TestModel",
                    "--latency_backend",
                    "analytical",
                    "--batching",
                    "paged-attn",
                    "--block_size",
                    "4",
                    "--qps",
                    "1",
                    "--distribution",
                    "burst",
                    "--request_count",
                    "2",
                    "--prefill_mean_len",
                    "8",
                    "--decode_mean_len",
                    "2",
                    "--prefill_range_len",
                    "0",
                    "--decode_range_len",
                    "0",
                    "--verbose",
                    "none",
                    "--results_path",
                    str(path / "results"),
                ]
            )
            args.af_compute_provider = FixedComputeProvider(0.001, 0.001)
            args.af_transfer_provider = FixedTransferProvider(0.001, 0.001)
            with (
                patch(
                    "TokenSim.cli.run.HardwareContext.load",
                    return_value=af_test_hardware(),
                ),
                redirect_stdout(io.StringIO()),
            ):
                engine = main(args)
            result = json.loads(
                next((path / "results").glob("result_*.json")).read_text()
            )
            self.assertTrue(result["af_enabled"])
            self.assertGreater(result["af_iteration_count"], 0)
            self.assertIsNotNone(result["af_runtime_state"])
            self.assertEqual(result["notdone"], [])
            self.assertEqual(len(engine.timing_recorder.timing.request_time), 2)
