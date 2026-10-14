from __future__ import annotations

import unittest
from tests.test_af_executor import af_test_resources
from dataclasses import replace

import simpy

from tests.test_af_parallel_planning import _cluster
from tests.test_af_runtime_state import _moe_config, _runtime_plan
from TokenSim.af.executor import AFExecutor
from TokenSim.af.latency import AFServiceEstimate
from TokenSim.af.models import AFBlock
from TokenSim.af.planning import ExecutionNode, RankLocalWork
from TokenSim.errors import ConfigurationError


class RecordingComputeProvider:
    def __init__(self, *, rejection: str | None = None) -> None:
        self.rejection = rejection
        self.validated = []
        self.queries = []

    def validate_effective_dtypes(self, model_id, effective_dtypes) -> None:
        self.validated.append((model_id, effective_dtypes))
        if self.rejection is not None:
            raise ValueError(self.rejection)

    def estimate(self, query) -> AFServiceEstimate:
        self.queries.append(query)
        return AFServiceEstimate(0.0, "dtype-contract:compute")


class RecordingTransferProvider:
    def __init__(self) -> None:
        self.queries = []

    def estimate(self, query) -> AFServiceEstimate:
        self.queries.append(query)
        return AFServiceEstimate(
            0.0,
            "dtype-contract:transfer",
            payload_bytes=query.payload_bytes,
        )


def _typed_runtime():
    cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
    cluster.af = replace(
        cluster.af,
        weight_dtype="fp8",
        activation_dtype="bf16",
        kv_cache_dtype="fp32",
    )
    cluster.af.links = [
        replace(
            cluster.af.links[0],
            a_to_f_dtype="fp8",
            f_to_a_dtype="fp32",
        )
    ]
    moe_config = _moe_config()
    plan = _runtime_plan(cluster, moe_config=moe_config, dtype="fp16")
    return cluster, moe_config, plan


def _executor(env, cluster, moe_config, plan, compute, transfer):
    return AFExecutor(
        env=env,
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        compute_provider=compute,
        transfer_provider=transfer,
        model_id="TestModel",
        # This compatibility input intentionally differs from the runtime plan.
        dtype="fp16",
        total_layers=plan.total_layers,
        hidden_size=plan.hidden_size,
        moe_config=moe_config,
        runtime_plan=plan,
        resources=af_test_resources(env, cluster, cluster.workers()),
    )


class AFExecutorDTypeContractTest(unittest.TestCase):
    def test_effective_roles_cross_executor_provider_boundary(self):
        env = simpy.Environment()
        cluster, moe_config, plan = _typed_runtime()
        compute = RecordingComputeProvider()
        transfer = RecordingTransferProvider()
        executor = _executor(
            env, cluster, moe_config, SimpleRuntimePlanView(plan), compute, transfer
        )

        self.assertEqual(executor.dtype, "bf16")
        self.assertEqual(len(compute.validated), 1)
        model_id, validated = compute.validated[0]
        self.assertEqual(model_id, "TestModel")
        self.assertEqual(validated.weight_dtype, "fp8")
        self.assertEqual(validated.activation_dtype, "bf16")
        self.assertEqual(validated.kv_cache_dtype, "fp32")

        block = AFBlock(
            iteration_id=0,
            block_id=0,
            phase="decode",
            request_ids=(7,),
            request_count=1,
            token_count=1,
            context_lengths=(9,),
            source_a_worker_id=0,
            link_name="af-link",
        )
        a_binding = plan.binding("a-domain", 0)
        f_binding = plan.binding("f-domain", 0)
        # Runtime-node validity is covered by runtime tests. This focused test
        # drives only the executor-to-provider query boundary.
        compute_node = ExecutionNode(
            node_id="attention",
            operation="attention",
            kind="compute",
            predecessor_ids=(),
            stage="attention",
            direction=None,
            source_domain="a-domain",
            target_domain="a-domain",
            logical_participants=(a_binding.instance,),
            domain_ranks=(a_binding.domain_rank,),
            physical_workers=(a_binding.physical_identity,),
            dp_rank=0,
            tp_rank=0,
            local_work=RankLocalWork(
                token_count=1,
                original_token_count=1,
                output_token_count=1,
            ),
            layer_type="dense",
        )
        env.process(executor._run_planned_compute(block, 0, compute_node))
        for direction, source, target in (
            ("a_to_f", a_binding, f_binding),
            ("f_to_a", f_binding, a_binding),
        ):
            transfer_node = ExecutionNode(
                node_id=direction,
                operation=f"af_{direction}",
                kind="communication",
                predecessor_ids=(),
                stage=None,
                direction=direction,
                source_domain=source.domain,
                target_domain=target.domain,
                logical_participants=(source.instance, target.instance),
                domain_ranks=(source.domain_rank, target.domain_rank),
                physical_workers=(
                    source.physical_identity,
                    target.physical_identity,
                ),
                dp_rank=0,
                network="af-net",
                resource_group="af-fabric",
                resource_lane=direction,
                payload_bytes=16,
                layer_type="dense",
            )
            env.process(
                executor._run_planned_communication(
                    block,
                    0,
                    cluster.af.links[0],
                    transfer_node,
                )
            )
        env.run()

        self.assertTrue(compute.queries)
        for query in compute.queries:
            self.assertEqual(query.dtype, "bf16")
            self.assertEqual(query.weight_dtype, "fp8")
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.kv_cache_dtype, "fp32")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp32")

        directional = {
            direction: [
                query for query in transfer.queries if query.direction == direction
            ]
            for direction in ("a_to_f", "f_to_a")
        }
        self.assertTrue(directional["a_to_f"])
        self.assertTrue(directional["f_to_a"])
        self.assertTrue(all(query.dtype == "fp8" for query in directional["a_to_f"]))
        self.assertTrue(all(query.dtype == "fp32" for query in directional["f_to_a"]))
        for query in transfer.queries:
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp32")

    def test_manifest_mismatch_fails_before_event_loop_or_query(self):
        env = simpy.Environment()
        cluster, moe_config, plan = _typed_runtime()
        compute = RecordingComputeProvider(
            rejection=(
                "operator manifest dtype mismatch: model='TestModel', "
                "role='activation_dtype', effective='bf16', manifest='fp16'"
            )
        )
        transfer = RecordingTransferProvider()

        with self.assertRaisesRegex(
            ConfigurationError,
            "provider dtype contract validation failed.*activation_dtype",
        ):
            _executor(env, cluster, moe_config, plan, compute, transfer)

        self.assertEqual(env.now, 0)
        self.assertEqual(compute.queries, [])
        self.assertEqual(transfer.queries, [])


class SimpleRuntimePlanView:
    def __init__(self, plan) -> None:
        self.total_layers = plan.total_layers
        self.hidden_size = plan.hidden_size
        self.effective_dtypes = plan.effective_dtypes
        self.link_effective_dtypes = plan.link_effective_dtypes

    def dtypes_for_link(self, link_name):
        for name, effective in self.link_effective_dtypes:
            if name == link_name:
                return effective
        return self.effective_dtypes

    @staticmethod
    def validate_execution_node(_node) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
