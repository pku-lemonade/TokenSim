from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.moe.config import MoEModelConfig
from TokenSim.placement import BalancedLoadWorkerPool, DataParallelWorkerPool
from tests.af.support.config import _cluster
from tests.af.support.execution import af_test_resources, execution_plan


class WorkloadComputeProvider:
    def __init__(self):
        self.queries = []

    def estimate(self, query):
        self.queries.append(query)
        duration = {
            "a": 1.0,
            "router": 0.25,
            "expert": query.local_token_count * 0.1,
            "attention": 1.0,
            "dense_ffn": query.local_token_count * 0.1,
            "layout": 0.0,
            "residual": 0.0,
        }[query.stage]
        return AFServiceEstimate(duration, "workload-stub")


class TypedTransferProvider:
    def __init__(self):
        self.queries = []

    def estimate(self, query):
        self.queries.append(query)
        duration = {
            "tp_collective": 0.5,
            "af_a_to_f": 0.5,
            "ep_dispatch": 0.25,
            "ep_combine": 0.25,
            "af_f_to_a": 0.5,
        }[query.operation]
        return AFServiceEstimate(
            duration,
            "typed-transfer-stub",
            payload_bytes=query.payload_bytes,
            details={"payload_bytes": query.payload_bytes},
        )


class ReplicaSkewComputeProvider:
    def estimate(self, query):
        if query.stage == "attention":
            duration = 2.0 if query.dp_rank == 0 else 0.1
        else:
            duration = 0.0
        return AFServiceEstimate(duration, "replica-skew-stub")


class ZeroTransferProvider:
    def estimate(self, query):
        return AFServiceEstimate(
            0.0, "zero-transfer-stub", payload_bytes=query.payload_bytes
        )


def _request(request_id, *, prefill_tokens=12, histogram=None):
    return SimpleNamespace(
        id=request_id,
        is_prefill=True,
        prefill_len=prefill_tokens,
        prefill_compute_len=prefill_tokens,
        context_len=prefill_tokens,
        prefill_expert_histogram=histogram,
        af_a_owners={},
    )


def _executor(env):
    cluster = _cluster()
    compute = WorkloadComputeProvider()
    transfer = TypedTransferProvider()
    moe = MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=1,
        num_moe_layers=1,
    )
    executor = AFExecutor(
        env,
        cluster,
        cluster.workers(),
        compute,
        transfer,
        "TestModel",
        "bf16",
        total_layers=1,
        hidden_size=16,
        runtime_plan=execution_plan(cluster),
        moe_config=moe,
        resources=af_test_resources(env, cluster, cluster.workers()),
    )
    return executor, compute, transfer


class AFParallelExecutionOracleTest(unittest.TestCase):
    def test_af_worker_pool_uses_actual_dp_ranks_for_global_round_robin(self):
        workers = [
            SimpleNamespace(
                id=rank * 2,
                dp_rank=rank,
                scheduler=SimpleNamespace(running=[], waiting=[]),
                workload=lambda: 0.0,
                cpu_workload=lambda: 0.0,
            )
            for rank in range(2)
        ]
        pool = LLMEngine._build_worker_pool(
            SimpleNamespace(), workers, BalancedLoadWorkerPool
        )
        self.assertIsInstance(pool, DataParallelWorkerPool)
        self.assertEqual([pool.schedule().dp_rank for _ in range(4)], [0, 1, 0, 1])

    def test_dp_placement_is_deterministic_and_a_owner_cannot_move(self):
        env = simpy.Environment()
        executor, _, _ = _executor(env)
        requests = [_request(index, prefill_tokens=1) for index in range(4)]
        first = executor.submit(0, requests, "prefill")
        self.assertEqual(
            [block.placement.f_instance for block in first.blocks],
            ["f0", "f1", "f0", "f1"],
        )
        self.assertTrue(
            all(block.placement.a_instance == "a0" for block in first.blocks)
        )
        with self.assertRaisesRegex(
            ConfigurationError, "cannot move.*without KV transfer"
        ):
            executor.submit(2, [requests[0]], "prefill")

        requests[0].is_prefill = False
        with self.assertRaisesRegex(
            ConfigurationError, "cannot move.*without KV transfer"
        ):
            executor.submit(2, [requests[0]], "decode")

        other, _, _ = _executor(simpy.Environment())
        replay = other.submit(
            0, [_request(index + 10, prefill_tokens=1) for index in range(4)], "prefill"
        )
        self.assertEqual(
            [block.placement for block in first.blocks],
            [block.placement for block in replay.blocks],
        )

    def test_dp_replicas_execute_independently(self):
        env = simpy.Environment()
        cluster = _cluster()
        moe = MoEModelConfig(
            is_moe_model=True,
            num_experts=4,
            num_experts_per_tok=1,
            num_moe_layers=1,
        )
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            ReplicaSkewComputeProvider(),
            ZeroTransferProvider(),
            "TestModel",
            "bf16",
            total_layers=1,
            hidden_size=16,
            runtime_plan=execution_plan(cluster),
            moe_config=moe,
            resources=af_test_resources(env, cluster, cluster.workers()),
        )
        slow = executor.submit(0, [_request(20, prefill_tokens=1)], "prefill")
        fast = executor.submit(2, [_request(21, prefill_tokens=1)], "prefill")
        env.process(executor.run_iteration(slow))
        env.process(executor.run_iteration(fast))
        env.run()

        self.assertAlmostEqual(slow.makespan, 2.0)
        self.assertAlmostEqual(fast.makespan, 0.1)
        fast_a = next(
            event
            for event in executor.trace.events_for(fast.iteration_id)
            if event.operation == "attention"
        )
        slow_a = next(
            event
            for event in executor.trace.events_for(slow.iteration_id)
            if event.operation == "attention"
        )
        self.assertLess(fast_a.end_at, slow_a.end_at)

    def test_missing_collective_fabric_alias_fails_before_events(self):
        env = simpy.Environment()
        cluster = _cluster()
        cluster.execution_domains["a-domain"].communication_plan.append(
            DomainCollectiveConfig(
                operation="all_reduce",
                after_stage="a_compute",
                participants=(0, 1),
                resource_group="missing-fabric",
            )
        )
        with self.assertRaisesRegex(ConfigurationError, "missing fabric alias"):
            AFExecutor(
                env,
                cluster,
                cluster.workers(),
                WorkloadComputeProvider(),
                TypedTransferProvider(),
                "TestModel",
                "bf16",
                total_layers=1,
                hidden_size=16,
                resources=af_test_resources(env, cluster, cluster.workers()),
            )
        self.assertEqual(env.now, 0)


if __name__ == "__main__":
    unittest.main()
