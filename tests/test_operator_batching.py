from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.af.latency import AFRoutingEdge
from TokenSim.af.planning import ExecutionNode, RankLocalWork, group_routed_expert_nodes
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data import (
    Calibration, CalibrationRule, OperatorDataGenerator, OperatorDataLookup,
    OperatorDataPackage, MissingOperatorDataError, estimate_operator,
)
from TokenSim.operator_data.schema import validate_batch_key
from tests.test_operator_data_latency_provider import _fixture_inputs, _hardware, _key


class BatchTimingTest(unittest.TestCase):
    def test_ragged_roofline_counts_all_kv_but_one_fixed_overhead(self):
        key = _key("generation_attention", batch_size=2)
        del key["context_len"]
        key["context_lengths"] = "1,9"
        estimate = estimate_operator("generation_attention_batch", key, _hardware("a-device"), Calibration(2, 5, "test"))
        singles = [estimate_operator("generation_attention", _key("generation_attention", context_len=n), _hardware("a-device"))
                   for n in (1, 9)]
        self.assertEqual(estimate.kv_bytes, sum(value.kv_bytes for value in singles))
        self.assertEqual(estimate.flops, sum(value.flops for value in singles))
        self.assertAlmostEqual(estimate.latency_us, 5 + 2 * max(estimate.compute_time_us, estimate.memory_time_us))
        self.assertEqual(estimate.details["formula_inputs"]["context_lengths"], (1, 9))

    def test_group_roofline_reads_each_expert_and_adds_one_overhead(self):
        key = dict(_key("moe", num_tokens=3, top_k=1, num_experts=2),
                   expert_token_counts="1,2", execution_boundary="unweighted_partials")
        estimate = estimate_operator("moe_group", key, _hardware("a-device"), Calibration(1, 10, "group-test"))
        self.assertEqual(estimate.weight_bytes, 3 * 2 * 8 * 16 * 2)
        self.assertEqual(estimate.flops, 6 * 3 * 8 * 16)
        self.assertAlmostEqual(estimate.latency_us, 10 + max(estimate.compute_time_us, estimate.memory_time_us))
        for invalid in (dict(key, expert_token_counts="3"), dict(key, execution_boundary="weighted_sum")):
            with self.assertRaises(ValueError):
                validate_batch_key("moe_group", invalid)

    def test_measured_values_bypass_calibration_and_old_packages_fail_clearly(self):
        source, shapes, observations = _fixture_inputs()
        generator = OperatorDataGenerator("test", {"a-device": _hardware("a-device"), "f-device": _hardware("f-device")},
            [source], calibrations={"moe_group": Calibration(100, 1000, "test-overhead")})
        with TemporaryDirectory() as directory:
            generator.generate(directory, "batch", shapes, observations)
            package = OperatorDataPackage.load(directory)
            row = package.tables["moe_group"][0]
            key = {name: row[name] for name in shapes[-1]["key"]}
            result = OperatorDataLookup(package).lookup("moe_group", key)
            self.assertLess(result.latency_us, 1000)
            analysis = package.analysis_tables["moe_group"][0]
            self.assertFalse(analysis["calibration_applied_to_selected"])
            self.assertIsNone(analysis["selected_fixed_overhead_us"])
            # Six-table v2 packages still load, but cannot silently emulate groups.
            for name in ("moe_group_perf.parquet", "moe_group_analysis.parquet"):
                (Path(directory) / name).unlink()
            old = OperatorDataPackage.load(directory)
            with self.assertRaisesRegex(MissingOperatorDataError, "regenerate"):
                OperatorDataLookup(old).lookup("moe_group", key)

    def test_calibration_is_shape_scoped_and_rejects_invalid_or_ambiguous_values(self):
        source, _, _ = _fixture_inputs()
        rule = CalibrationRule("gemm", {"device": "a-device", "n": 24, "k": 8}, Calibration(1, 3, "qkv"))
        generator = OperatorDataGenerator("test", {"a-device": _hardware("a-device")}, [source], shape_calibrations=[rule])
        self.assertEqual(generator._calibration_for("gemm", _key("gemm")).offset_us, 3)
        self.assertIsNone(generator._calibration_for("gemm", _key("gemm", n=8)))
        generator.shape_calibrations = (rule, rule)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            generator._calibration_for("gemm", _key("gemm"))
        for multiplier, offset in ((0, 0), (-1, 0), (1, -1), (float("inf"), 0), (1, float("nan"))):
            with self.assertRaises(ValueError):
                Calibration(multiplier, offset)


class PhysicalExpertGroupingTest(unittest.TestCase):
    def expert(self, expert, *, worker="f:0", rank=0, tp_group=(0,), prefix="i0:b0:l0"):
        edge = AFRoutingEdge(request_id=0, token_id=0, expert_id=expert, layer_id=0, phase="decode",
                             iteration_id=0, decode_step=0, a_dp_rank=0, a_tp_rank=0,
                             top_k_index=expert, routing_weight=0.5)
        work = RankLocalWork(token_count=1, tensor_shape=(("tokens", 1), ("hidden_size", 8)),
            expert_ids=(expert,), expert_workload=((expert, 1),), original_token_count=1,
            assignment_count=1, output_token_count=1, routing_edges=(edge,))
        return ExecutionNode(node_id=f"{prefix}:routed:e{expert}:r{rank}", operation="routed_expert",
            kind="compute", predecessor_ids=(), stage="routed_expert", direction=None,
            source_domain="f", target_domain="f", logical_participants=("f0",),
            domain_ranks=(rank,), physical_workers=(worker,), dp_rank=0,
            tp_rank=rank, tp_group=tp_group, local_work=work, routing_edges=(edge,), layer_type="moe")

    def test_group_preserves_assignments_and_redirects_all_consumers(self):
        first, second = self.expert(0), self.expert(1)
        consumer = replace(first, node_id="combine", stage="weighted_combine",
                           predecessor_ids=(first.node_id, second.node_id))
        grouped = group_routed_expert_nodes((first, second, consumer))
        self.assertEqual(len(grouped), 2)
        work = grouped[0].local_work
        self.assertEqual(work.expert_ids, (0, 1))
        self.assertEqual(work.assignment_count, 2)
        self.assertEqual(work.original_token_count, 1)
        self.assertEqual(set(work.routing_edges), set(first.routing_edges + second.routing_edges))
        self.assertEqual(grouped[1].predecessor_ids, (grouped[0].node_id,))

    def test_different_gpu_tp_layer_and_dependent_nodes_never_merge(self):
        first = self.expert(0)
        for second in (self.expert(1, worker="f:1"), self.expert(1, tp_group=(0, 1)),
                       self.expert(1, prefix="i0:b0:l1"), self.expert(1, prefix="i1:b0:l0")):
            self.assertEqual(len(group_routed_expert_nodes((first, second))), 2)
        dependent = replace(self.expert(1), predecessor_ids=(first.node_id,))
        with self.assertRaisesRegex(ConfigurationError, "dependent"):
            group_routed_expert_nodes((first, dependent))


if __name__ == "__main__":
    unittest.main()
