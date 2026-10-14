"""Af runtime state."""

from __future__ import annotations

import json
import unittest

from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.errors import SimulationStateError
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.reporting import get_af_stats
from TokenSim.workload.source import LLMSource
from tests.support.af_parallel import _cluster
from tests.support.af_runtime import (
    _engine,
    _moe_config,
    _request,
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
        env, engine = _engine(cluster, moe_config=_moe_config())
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
