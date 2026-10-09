"""Operator holdout oracles and nonuniform/full-input ownership guards."""

from copy import deepcopy
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from TokenSim.llm.forward import ForwardPass, RankShape
from TokenSim.latency.moe_efficiency import ExpertEfficiencyTransfer
from TokenSim.latency.moe_work import MegaMoEWorkModel
from tests import test_full_mtp as full_mtp_tests


class MoEPhaseEfficiencyTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = HardwareContext.load("data")
        cls.model = cls.context.model("DeepSeek-V4-Pro")
        cls.links = LinkCatalog.load("data/topologies/links.yaml")
        cls.scope = {"dtype": "mxfp4", "kernel_source": "deepgemm_megamoe", "kernel_dtype": "fp8_fp4",
                     "distribution": "power_law_1.01", "pre_dispatch": "sglang_jit", "source_policy": "random",
                     "hidden_size": 7168, "inter_size": 3072, "top_k": 6, "num_experts": 384,
                     "num_fused_shared_experts": 0, "tp_size": 1, "ep_size": 8}

    def inputs(self, sku="b200"):
        device = self.context.device(sku)
        model = MegaMoEWorkModel(self.model, device, 8,
            nvlink_bytes_per_s=self.links.get(device.scale_up_link).bandwidth_per_direction_bytes_per_s)
        rows = [row for row in self.context.operator_package(device.device_id, "vllm").tables["dsv4_megamoe"]
                if all(row[key] == value for key, value in self.scope.items())]
        return model, deepcopy(rows)

    @staticmethod
    def uniform(model, rows):
        inputs = {rank: rows for rank in range(8)}
        return model.estimate(inputs, inputs)

    def test_all_uniform_original_anchors_are_preserved(self):
        for sku in ("b200", "b300"):
            model, rows = self.inputs(sku)
            before = deepcopy(rows)
            transfer = ExpertEfficiencyTransfer(model, rows)
            for row in rows:
                with self.subTest(sku=sku, rows=row["num_tokens"]):
                    work = self.uniform(model, row["num_tokens"])
                    result = transfer.price(row["latency_us"], work, work,
                        anchor_rows=row["num_tokens"], phase=row["phase"])
                    self.assertAlmostEqual(result.latency_us, row["latency_us"])
                    self.assertEqual(result.efficiency_ratio, 1)
            self.assertEqual(rows, before)

    def test_directed_independent_holdout_oracles_keep_full_common_work(self):
        fixture = json.loads(Path("data/reference/agentx-mtp-execution/phase-efficiency-holdout-followup.json").read_text())
        for sku, oracle in fixture["hardware"].items():
            model, rows = self.inputs(sku)
            for case in oracle["directed_holdouts"]:
                with self.subTest(sku=sku, anchor=case["anchor_rows"], heldout=case["heldout_rows"]):
                    training = [row for row in rows if not (row["phase"] == case["phase"]
                        and row["effective_num_max_tokens_per_rank"] == case["effective_capacity"]
                        and row["num_tokens"] == case["heldout_rows"])]
                    anchor = next(row for row in training if row["phase"] == case["phase"]
                                  and row["num_tokens"] == case["anchor_rows"])
                    result = ExpertEfficiencyTransfer(model, training).price(
                        anchor["latency_us"], self.uniform(model, case["anchor_rows"]),
                        self.uniform(model, case["heldout_rows"]), anchor_rows=case["anchor_rows"], phase=case["phase"])
                    self.assertAlmostEqual(result.common_us, case["full_input_phase_us_kept_at_anchor_efficiency"])
                    self.assertAlmostEqual(result.latency_us, case["predicted_us"])
                    self.assertLessEqual(abs(result.latency_us / case["measured_heldout_us"] - 1), .25)

    def test_zero_routed_work_keeps_full_input_and_startup_cost(self):
        model, rows = self.inputs()
        transfer = ExpertEfficiencyTransfer(model, rows)
        inputs = {rank: 64 for rank in range(8)}
        source = model.estimate(inputs, inputs)
        target = model.estimate(inputs, {rank: 0 for rank in inputs})
        anchor = next(row["latency_us"] for row in rows if row["phase"] == "generation" and row["num_tokens"] == 64)
        result = transfer.price(anchor, source, target, anchor_rows=64, phase="generation")
        self.assertGreater(result.common_us, 0)
        self.assertEqual(result.expert_us, 0)
        self.assertEqual(result.match, "zero_routed_expert_work")
        self.assertAlmostEqual(result.latency_us, anchor * target.latency_us / source.latency_us)

    def test_singleton_cross_capacity_and_cross_regime_retain_original_ratio(self):
        model, rows = self.inputs("b300")
        transfer = ExpertEfficiencyTransfer(model, rows)
        for phase, size in (("context", 1024), ("generation", 192), ("cross_buffer_regime", 600)):
            with self.subTest(phase=phase):
                source = self.uniform(model, size)
                target = model.estimate({rank: size for rank in range(8)}, {rank: 1 for rank in range(8)})
                result = transfer.price(100, source, target, anchor_rows=size, phase=phase)
                self.assertEqual(result.efficiency_ratio, 1)
                self.assertAlmostEqual(result.latency_us, 100 * target.latency_us / source.latency_us)

    def test_missing_capacity_metadata_cannot_become_a_curve(self):
        model, rows = self.inputs()
        for row in rows:
            row.pop("effective_num_max_tokens_per_rank")
        transfer = ExpertEfficiencyTransfer(model, rows)
        self.assertEqual(transfer.curves, {})

    def test_nonuniform_estimator_prices_only_full_anchor_and_preserves_shared(self):
        full_mtp_tests.FullMTPTest.setUpClass()
        backend = full_mtp_tests.FullMTPTest().backend()
        estimator = backend.native_moe
        estimator.set_step(ForwardPass({rank: RankShape(512 if rank == 0 else 16, None, True,
            1, valid_tokens=32 if rank == 0 else 16) for rank in range(8)}))
        with patch.object(estimator.lookup, "lookup", wraps=estimator.lookup.lookup) as queries:
            total, routed, shared = estimator.estimate(512)
        self.assertEqual(queries.call_count, 1)
        self.assertEqual(queries.call_args.args[1]["num_tokens"], 512)
        self.assertEqual(estimator.counts["full_input_rows"], 512)
        self.assertEqual(estimator.counts["routed_valid_rows"], 32)
        self.assertAlmostEqual(shared, estimator.shared_fused_delta(512))
        self.assertAlmostEqual(total, (routed + shared) * 61)
        self.assertAlmostEqual(estimator.composition["transferred_common_seconds"]
                               + estimator.composition["transferred_expert_seconds"], routed * 61 * 1e-6)
