"""Runtime state, KV partitions and physical memory budgets."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

from TokenSim.af.runtime.topology import kv_head_partitions
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.reporting import get_af_stats
from TokenSim.workload.source import LLMSource
from tests.af.support.config import _cluster
from tests.af.support.model import (
    _engine,
    _moe,
    _request,
    _runtime_hardware,
    _runtime_plan,
    _split_phase_alias_cluster,
)


class AFRuntimeStateRegressionTest(unittest.TestCase):
    def test_split_phase_domains_share_equivalent_physical_scheduler_state(self):
        cluster = _split_phase_alias_cluster()
        _, engine = _engine(cluster)

        self.assertEqual(len(engine.request_workers), 1)
        manager = engine.request_workers[0].scheduler.block_manager
        self.assertIs(
            engine.af_replica_block_managers[("prefill-a", 0)],
            manager,
        )
        self.assertIs(
            engine.af_replica_block_managers[("decode-a", 0)],
            manager,
        )
        request = _request(11)
        manager.allocate(request)
        self.assertEqual(
            request.af_a_owners,
            {"prefill-a": "a0", "decode-a": "a0-decode"},
        )
        summary = engine.af_runtime_summary()
        replica_usage = {
            item["domain"]: item["logical_gpu_usage"]
            for item in summary["replicas"]
            if item["domain"] in {"prefill-a", "decode-a"}
        }
        self.assertEqual(
            replica_usage["prefill-a"],
            replica_usage["decode-a"],
        )


class RuntimePlannerAndSummaryTest(unittest.TestCase):
    def test_runtime_summary_is_versioned_and_serializable(self):
        plan = _runtime_plan()
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        req = _request(9)
        manager.allocate(req)
        summary = plan.summary({replica.replica_key: manager})

        self.assertEqual(summary["schema"], "tokensim.af_runtime_state")
        self.assertEqual(summary["version"], "2.0")
        self.assertGreater(summary["replicas"][0]["logical_gpu_usage"]["used"], 0)
        self.assertEqual(summary["replicas"][0]["logical_cpu_usage"]["used"], 0)
        manager.swap_out(req)
        swapped = plan.summary({replica.replica_key: manager})
        self.assertGreater(swapped["replicas"][0]["logical_cpu_usage"]["used"], 0)
        rebuilt = json.loads(json.dumps(summary))
        self.assertEqual(len(rebuilt["rank_bindings"]), len(plan.rank_bindings))
        self.assertEqual(len(rebuilt["physical_workers"]), len(plan.physical_states))
        f_domain = next(
            item for item in rebuilt["domains"] if item["name"] == "f-domain"
        )
        self.assertEqual(f_domain["base_world_size"], 4)
        self.assertEqual(f_domain["expert_parallel_mode"], "rank_reuse")
        self.assertEqual(f_domain["expert_tensor_parallel_size"], 1)
        f_bindings = [
            item for item in rebuilt["rank_bindings"] if item["domain"] == "f-domain"
        ]
        self.assertEqual(
            {tuple(item["ep_group"]) for item in f_bindings},
            {(0, 1, 2, 3)},
        )
        for state in rebuilt["physical_workers"]:
            self.assertEqual(
                state["model_bytes"],
                sum(component["bytes"] for component in state["components"]),
            )
        self.assertEqual(
            rebuilt["effective_dtypes"]["bytes_per_element"]["weight_dtype"],
            2,
        )

    def test_dp_replicas_have_independent_cache_and_stable_request_owner(self):
        cluster = _cluster(a_tp=2, a_dp=2, f_tp=1, f_dp=1, f_ep=1)
        _, engine = _engine(cluster)
        managers = engine.af_replica_block_managers
        manager0 = managers[("a-domain", 0)]
        manager1 = managers[("a-domain", 1)]
        req0 = _request(30)
        req1 = _request(31)
        manager0.allocate(req0)
        manager1.allocate(req1)

        self.assertEqual(req0.af_a_owners, {"a-domain": "a0"})
        self.assertEqual(req1.af_a_owners, {"a-domain": "a1"})
        self.assertEqual(manager0.get_gpu_status()[1], 2)
        self.assertEqual(manager1.get_gpu_status()[1], 2)
        manager0.release_request_blocks(req0)
        self.assertEqual(manager0.get_gpu_status()[1], 0)
        self.assertEqual(manager1.get_gpu_status()[1], 2)

        req1.status = RequestStatus.RUNNING
        with self.assertRaisesRegex(SimulationStateError, "owner conflict"):
            manager0.allocate(req1)

    def test_combined_tp_dp_ep_engine_runs_with_one_scheduler_per_a_replica(self):
        cluster = _cluster(a_tp=2, a_dp=2, f_tp=2, f_dp=2, f_ep=4)
        env, engine = _engine(cluster, moe_config=_moe(top_k=1, shared=1))
        requests = [
            Request(
                request_id,
                4,
                1,
                block_size=4,
            )
            for request_id in (40, 41)
        ]
        env.process(LLMSource(env, engine, requests, qps=1, distribution="burst"))

        # Use the public task enum without reaching through the engine instance.
        from TokenSim.llm.llm_engine import Task

        def stop_when_done():
            while env.now < 1:
                if all(request.is_done for request in requests):
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return
                yield env.timeout(1e-5)
            raise AssertionError("combined AF requests did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)

        self.assertEqual(len(engine.request_workers), 2)
        self.assertEqual({worker.dp_rank for worker in engine.request_workers}, {0, 1})
        self.assertTrue(all(request.is_done for request in requests))
        self.assertEqual(
            {tuple(request.af_a_owners.items()) for request in requests},
            {(("a-domain", "a0"),), (("a-domain", "a1"),)},
        )
        f_workers = [
            worker
            for worker in engine.workers
            if any(
                binding.capability == "af_ffn" for binding in worker.af_runtime_bindings
            )
        ]
        self.assertEqual(len(f_workers), 4)
        self.assertTrue(any(3 in worker.owned_expert_ids for worker in f_workers))
        operations = [event.operation for event in engine.af_executor.trace.events]
        for operation in ("af_a_to_f", "af_f_to_a"):
            stages = {
                event.logical_stage_id
                for event in engine.af_executor.trace.events
                if event.operation == operation
            }
            self.assertEqual(len(stages), 4)
        runtime = get_af_stats(engine)["af_runtime_state"]
        self.assertEqual(runtime["schema"], "tokensim.af_runtime_state")


class AFRuntimePlanTest(unittest.TestCase):
    def test_kv_heads_shard_replicate_and_reject_ambiguous_topology(self):
        sharded = kv_head_partitions(
            num_kv_heads=8,
            tensor_parallel_size=2,
            head_dim=2,
            num_layers=2,
            block_size=4,
            dtype_size_bytes=2,
        )
        replicated = kv_head_partitions(
            num_kv_heads=1,
            tensor_parallel_size=2,
            head_dim=2,
            num_layers=2,
            block_size=4,
            dtype_size_bytes=2,
        )

        self.assertEqual(
            [item.global_head_ids for item in sharded], [(0, 1, 2, 3), (4, 5, 6, 7)]
        )
        self.assertEqual([item.replica_factor for item in sharded], [1, 1])
        self.assertEqual([item.global_head_ids for item in replicated], [(0,), (0,)])
        self.assertEqual([item.replica_factor for item in replicated], [2, 2])
        with self.assertRaisesRegex(ConfigurationError, "cannot be evenly sharded"):
            kv_head_partitions(
                num_kv_heads=3,
                tensor_parallel_size=2,
                head_dim=2,
                num_layers=2,
                block_size=4,
                dtype_size_bytes=2,
            )

    def test_component_ownership_matches_hand_calculation(self):
        plan = _runtime_plan()
        a_leader = plan.physical_state("a-pool:0")
        a_follower = plan.physical_state("a-pool:1")
        f_ep0 = plan.physical_state("f-pool:0")
        f_ep1 = plan.physical_state("f-pool:2")

        self.assertEqual(a_leader.component_bytes()["attention"], 2048)
        self.assertEqual(a_leader.component_bytes()["embedding_lm_head"], 1600000)
        self.assertEqual(a_leader.component_bytes()["router"], 128)
        self.assertNotIn("router", a_follower.component_bytes())
        self.assertNotIn("routed_expert", a_leader.component_bytes())
        self.assertNotIn("router", f_ep0.component_bytes())
        self.assertEqual(f_ep0.component_bytes()["dense_ffn"], 3072)
        self.assertEqual(f_ep0.component_bytes()["routed_expert"], 3072)
        self.assertEqual(f_ep0.component_bytes()["shared_expert"], 1536)
        self.assertNotIn("router", f_ep1.component_bytes())
        self.assertEqual(f_ep1.component_bytes()["dense_ffn"], 3072)
        self.assertEqual(f_ep1.component_bytes()["routed_expert"], 3072)
        self.assertEqual(f_ep1.component_bytes()["shared_expert"], 1536)
        self.assertEqual(f_ep0.component_bytes(), f_ep1.component_bytes())

        a_bindings = [
            binding
            for binding in plan.rank_bindings
            if binding.capability == "af_attention"
        ]
        for binding in a_bindings:
            state = plan.physical_state(binding.physical_identity)
            if binding.is_scheduler_leader:
                self.assertIn("router", state.component_bytes())
            else:
                self.assertNotIn("router", state.component_bytes())
        for binding in plan.rank_bindings:
            if binding.capability == "af_ffn":
                self.assertNotIn(
                    "router",
                    plan.physical_state(binding.physical_identity).component_bytes(),
                )

        self.assertEqual([layer.layer_type for layer in plan.layers], ["moe", "dense"])
        for state in plan.physical_states:
            for component in state.components:
                if component.layer_id == 0:
                    self.assertNotEqual(component.category, "dense_ffn")
                if component.layer_id == 1:
                    self.assertNotIn(
                        component.category,
                        {"router", "routed_expert", "shared_expert"},
                    )
            self.assertEqual(
                len(state.components),
                len({component.component_id for component in state.components}),
            )

    def test_attention_memory_uses_actual_mha_gqa_and_mqa_widths(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        expected_bytes = {8: 1024, 2: 640, 1: 640}
        for kv_heads, expected in expected_bytes.items():
            with self.subTest(kv_heads=kv_heads):
                plan = _runtime_plan(
                    cluster,
                    hardware=_runtime_hardware(layers=1),
                    moe_config=replace(
                        _moe(top_k=1, shared=1),
                        num_moe_layers=1,
                        num_key_value_heads=kv_heads,
                    ),
                )
                state = plan.physical_state("a-pool:0")
                attention = next(
                    component
                    for component in state.components
                    if component.category == "attention"
                )
                self.assertEqual(attention.bytes, expected)
                self.assertEqual(
                    dict(attention.shape)["kv_heads_local"], 4 if kv_heads == 8 else 1
                )
                self.assertIn("Q/K/V/O", attention.formula)

    def test_weight_and_kv_dtypes_change_independent_ledgers(self):
        baseline_cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        typed_cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        typed_cluster.af = replace(
            typed_cluster.af,
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp32",
        )
        typed_cluster.af.links = [
            replace(
                typed_cluster.af.links[0],
                a_to_f_dtype="fp8",
                f_to_a_dtype="fp16",
            )
        ]

        baseline = _runtime_plan(baseline_cluster, dtype="bf16")
        typed = _runtime_plan(typed_cluster, dtype="bf16")
        baseline_a = baseline.physical_state("a-pool:0")
        typed_a = typed.physical_state("a-pool:0")
        baseline_binding = baseline.binding("a-domain", 0)
        typed_binding = typed.binding("a-domain", 0)

        self.assertEqual(typed.effective_dtypes.weight_dtype, "fp8")
        self.assertEqual(typed.effective_dtypes.activation_dtype, "bf16")
        self.assertEqual(typed.effective_dtypes.kv_cache_dtype, "fp32")
        self.assertEqual(typed.effective_dtypes.a_to_f_dtype, "fp8")
        self.assertEqual(typed.effective_dtypes.f_to_a_dtype, "fp16")
        self.assertEqual(typed_a.model_bytes * 2, baseline_a.model_bytes)
        self.assertEqual(
            typed_binding.kv_partition.bytes_per_block,
            baseline_binding.kv_partition.bytes_per_block * 2,
        )

    def test_oom_and_missing_metadata_fail_before_execution(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            (
                "domain/rank=.*physical_worker=.*component_bytes=.*model_bytes=.*"
                "capacity_bytes=.*required_bytes=.*available_bytes"
            ),
        ):
            _runtime_plan(hardware=_runtime_hardware(capacity=0.000001))

        unknown_hardware = _cluster()
        for group in unknown_hardware.worker_groups:
            group.hardware = "missing-device"
        with self.assertRaisesRegex(
            ConfigurationError, "unknown hardware 'missing-device'"
        ):
            _runtime_plan(unknown_hardware)
        with self.assertRaisesRegex(ConfigurationError, "no explicit byte size"):
            _runtime_plan(dtype="mystery")
