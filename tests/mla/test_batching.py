from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from TokenSim.af.planning.grouping import group_routed_expert_nodes
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.providers.queries import AFRoutingEdge
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorResolver
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.mla.test_attention import core_key


class BatchTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = HardwareContext.load("data").device("a100_sxm_80g")
        cls.analytical = analytical_model_for(cls.device.family)

    def estimate(self, table, key, calibration=None):
        return self.analytical.estimate(table, key, self.device, calibration)

    def test_ragged_roofline_counts_all_kv_but_one_launch(self):
        batch = self.estimate(
            "mla_attention",
            core_key(batch_size=2, context_lengths="[[1,0.5],[9,0.5]]"),
            Calibration(2, 5),
        )
        singles = [
            self.estimate("mla_attention", core_key(context_lengths=f"[[{n},1]]"))
            for n in (1, 9)
        ]
        self.assertEqual(batch.work.kv_bytes, sum(x.work.kv_bytes for x in singles))
        self.assertEqual(batch.work.flops, sum(x.work.flops for x in singles))
        self.assertEqual(batch.overhead_us, singles[0].overhead_us)
        self.assertAlmostEqual(batch.latency_us, 2 * batch.roofline_us + 5)

    def test_group_reads_each_expert_with_current_three_kernel_model(self):
        key = dict(dtype="bf16", hidden_size=8, inter_size=16, expert_tokens="[1,2]")
        estimate = self.estimate("routed_experts", key, Calibration(1, 10))
        self.assertEqual(estimate.work.weight_bytes, 3 * 2 * 8 * 16 * 2)
        self.assertEqual(estimate.work.flops, 6 * 3 * 8 * 16 + 4 * 3 * 16)
        self.assertEqual(
            estimate.overhead_us, 3 * self.device.analytical.get("kernel_launch_us", 4)
        )
        self.assertAlmostEqual(estimate.latency_us, estimate.roofline_us + 10)

    def package(self):
        source = SourceRecord("fixture", "A", "measured")
        key = dict(dtype="bf16", hidden_size=8, inter_size=16, expert_tokens="[1,2]")
        meta = PackageMeta(
            "fixture-v1",
            self.device.device_id,
            "test",
            sources={source.source_id: source},
            calibration={
                "routed_experts": Calibration(100, 1000, "group").to_dict(),
                "default": Calibration(2, 3, "default").to_dict(),
            },
        )
        return (
            OperatorDataPackage.from_rows(
                meta,
                {
                    "routed_experts": [
                        {**key, "latency_us": 7.0, "source_id": source.source_id}
                    ]
                },
            ),
            key,
        )

    def test_measured_bypasses_calibration_roundtrip_and_missing_table_fails(self):
        package, key = self.package()
        with TemporaryDirectory() as directory:
            package.write(directory)
            loaded = OperatorDataPackage.load(directory)
            resolver = OperatorResolver(device=self.device, package=loaded)
            result = resolver.query("routed_experts", key)
            self.assertEqual(
                (result.latency_us, result.source_id, result.match_type),
                (7.0, "fixture", "exact"),
            )
            missing = replace(loaded, tables={})
            with self.assertRaises(MissingOperatorDataError):
                OperatorLookup(missing).lookup("routed_experts", key)
            with self.assertRaises(MissingOperatorDataError):
                OperatorLookup(loaded, LookupPolicy(extrapolate="none")).lookup(
                    "routed_experts", dict(key, expert_tokens="[3]")
                )

    def test_table_calibration_isolated_from_default_and_invalid_values_rejected(self):
        # Shape-scoped CalibrationRule is retired; current packages select table/default.
        package, key = self.package()
        resolver = OperatorResolver(device=self.device, package=package)
        grouped = dict(key, expert_tokens="[3]")
        self.assertAlmostEqual(
            resolver.query("routed_experts", grouped).latency_us,
            self.estimate("routed_experts", grouped).latency_us * 100 + 1000,
        )
        gemm = dict(dtype="bf16", m=1, n=24, k=8)
        self.assertAlmostEqual(
            resolver.query("gemm", gemm).latency_us,
            self.estimate("gemm", gemm).latency_us * 2 + 3,
        )
        for multiplier, offset in (
            (0, 0),
            (-1, 0),
            (1, -1),
            (float("inf"), 0),
            (1, float("nan")),
            (True, 0),
            (1, False),
        ):
            with self.subTest(multiplier=multiplier, offset=offset):
                with self.assertRaises(ConfigurationError):
                    Calibration(multiplier, offset)
                with self.assertRaises(ConfigurationError):
                    Calibration.from_mapping(
                        dict(multiplier=multiplier, offset_us=offset)
                    )


class PhysicalExpertGroupingTest(unittest.TestCase):
    def expert(self, expert, *, worker="f:0", rank=0, tp_group=(0,), prefix="i0:b0:l0"):
        edge = AFRoutingEdge(
            request_id=0,
            token_id=0,
            expert_id=expert,
            layer_id=0,
            phase="decode",
            iteration_id=0,
            decode_step=0,
            a_dp_rank=0,
            a_tp_rank=0,
            top_k_index=expert,
            routing_weight=0.5,
        )
        work = RankLocalWork(
            token_count=1,
            tensor_shape=(("tokens", 1), ("hidden_size", 8)),
            expert_ids=(expert,),
            expert_workload=((expert, 1),),
            original_token_count=1,
            assignment_count=1,
            output_token_count=1,
            routing_edges=(edge,),
        )
        return ExecutionNode(
            node_id=f"{prefix}:routed:e{expert}:r{rank}",
            operation="routed_expert",
            kind="compute",
            predecessor_ids=(),
            stage="routed_expert",
            direction=None,
            source_domain="f",
            target_domain="f",
            logical_participants=("f0",),
            domain_ranks=(rank,),
            physical_workers=(worker,),
            dp_rank=0,
            tp_rank=rank,
            tp_group=tp_group,
            local_work=work,
            routing_edges=(edge,),
            layer_type="moe",
        )

    def test_group_preserves_assignments_and_redirects_all_consumers(self):
        first, second = self.expert(0), self.expert(1)
        consumer = replace(
            first,
            node_id="combine",
            stage="weighted_combine",
            predecessor_ids=(first.node_id, second.node_id),
        )
        grouped = group_routed_expert_nodes((first, second, consumer))
        self.assertEqual(len(grouped), 2)
        work = grouped[0].local_work
        self.assertEqual(work.expert_ids, (0, 1))
        self.assertEqual(work.assignment_count, 2)
        self.assertEqual(work.original_token_count, 1)
        self.assertEqual(
            set(work.routing_edges), set(first.routing_edges + second.routing_edges)
        )
        self.assertEqual(grouped[1].predecessor_ids, (grouped[0].node_id,))

    def test_different_gpu_tp_layer_and_dependent_nodes_never_merge(self):
        first = self.expert(0)
        for second in (
            self.expert(1, worker="f:1"),
            self.expert(1, tp_group=(0, 1)),
            self.expert(1, prefix="i0:b0:l1"),
            self.expert(1, prefix="i1:b0:l0"),
        ):
            self.assertEqual(len(group_routed_expert_nodes((first, second))), 2)
        dependent = replace(self.expert(1), predecessor_ids=(first.node_id,))
        with self.assertRaisesRegex(ConfigurationError, "dependent"):
            group_routed_expert_nodes((first, dependent))


if __name__ == "__main__":
    unittest.main()
