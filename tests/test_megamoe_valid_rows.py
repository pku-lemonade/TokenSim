"""Source-derived full-input, valid-row, SP, dummy and cost ownership."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from TokenSim.latency.deepseek_v4_moe import MoEPass, RankShape, sp_valid_rows
from TokenSim.latency.deepseek_v4_moe_work import MegaMoEWorkModel, block_m_for_input
from util.results import get_latency_stats
import tests.test_native_megamoe as native_moe
import tests.test_full_mtp as full_mtp


class MegaMoEValidRowsTest(TestCase):
    @classmethod
    def setUpClass(cls):
        native_moe.NativeMegaMoETest.setUpClass()
        full_mtp.FullMTPTest.setUpClass()

    def backend(self, **kwargs):
        return native_moe.NativeMegaMoETest().backend(**kwargs)

    def test_graph_inputs_are_full_while_valid_routes_keep_real_rows(self):
        backend = self.backend()
        backend.set_moe_step(MoEPass({rank: RankShape(n, 4, False, n // 4)
                                     for rank, n in enumerate([64] + [16] * 7)}))
        group = backend.native_moe.group_rows(64)
        self.assertTrue(group.graph)
        self.assertEqual(list(group.inputs.values()), [64] * 8)
        self.assertEqual(sum(group.routed.values()), 176)
        self.assertEqual(group.routed[0], 64)
        self.assertEqual(group.routed[1], 16)

    def test_sp_contiguous_valid_slices_and_added_tail_are_not_averaged(self):
        self.assertEqual([sp_valid_rows(20, 32, 4, rank) for rank in range(4)], [8, 8, 4, 0])
        self.assertEqual([sp_valid_rows(20, 20, 4, rank) for rank in range(4)], [5, 5, 5, 5])
        self.assertEqual([sp_valid_rows(19, 19, 4, rank) for rank in range(4)], [5, 5, 5, 4])
        backend = self.backend(tp=4, dp=2)
        backend.set_moe_step(MoEPass({0: RankShape(32, 4, False, 8), 1: RankShape(20, 4, False, 5)}))
        group = backend.native_moe.group_rows(32)
        self.assertEqual(list(group.inputs.values()), [8] * 8)
        self.assertEqual(list(group.routed.values()), [8, 8, 8, 8, 8, 8, 4, 0])
        self.assertEqual(sum(group.routed.values()), 52)

    def test_dummy_and_mtp_passes_preserve_zero_routed_requests(self):
        backend = full_mtp.FullMTPTest().backend()
        step = MoEPass({0: RankShape(16, 4, False, 4, 16, 4),
                       **{rank: RankShape(4, 4, False, 1, 0, 0) for rank in range(1, 8)}})
        backend.set_moe_step(step)
        initial, later, final = backend.native_mtp.pass_shapes(full_mtp.FullMTPTest().decode_requests(4))
        self.assertIs(initial[2], step)
        self.assertEqual(later[2].ranks[1].tokens, 1)  # dummy forward stays present
        self.assertEqual(later[2].ranks[1].routed_tokens, 0)
        self.assertEqual(final[2].ranks[1].valid_requests, 0)
        backend.native_mtp.decode_moe.set_step(later[2])
        group = backend.native_mtp.decode_moe.group_rows(later[0])
        self.assertEqual(set(group.inputs.values()), {4})
        self.assertEqual(sum(group.routed.values()), 4)

    def test_invalid_valid_row_and_request_counts_are_rejected(self):
        for kwargs in ({"valid_tokens": -1}, {"valid_tokens": 17}, {"valid_requests": -1}, {"valid_requests": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                RankShape(16, 4, False, 4, **kwargs)

    def test_uniform_measured_module_remains_exact_without_ep_multiplication(self):
        backend = self.backend()
        for rows in (1, 64, 128, 1024):
            with self.subTest(rows=rows):
                backend.set_moe_step(MoEPass({rank: RankShape(rows, 1, True) for rank in range(8)}))
                total, routed, shared = backend.native_moe.estimate(rows, layers=1)
                self.assertEqual(routed, 100 + rows)
                self.assertAlmostEqual(total, routed + shared)

    def test_masking_changes_routed_work_and_preserves_full_shared_input(self):
        backend = self.backend()
        full = MoEPass({rank: RankShape(128, 4, False, 32) for rank in range(8)})
        masked = MoEPass({rank: RankShape(128, 4, False, 32, 16, 4) for rank in range(8)})
        backend.set_moe_step(full)
        _, before, shared_before = backend.native_moe.estimate(128)
        backend.set_moe_step(masked)
        _, after, shared_after = backend.native_moe.estimate(128)
        self.assertLess(after, before)
        self.assertEqual(shared_before, shared_after)
        group = backend.native_moe.group_rows(128)
        self.assertEqual(set(group.inputs.values()), {128})
        self.assertEqual(sum(group.routed.values()), 128)

    def test_all_padding_retains_kernel_scans_and_collective_progress(self):
        backend = self.backend()
        backend.set_moe_step(MoEPass({rank: RankShape(4, 4, False, 1, 0, 0) for rank in range(8)}))
        total, routed, shared = backend.native_moe.estimate(4, layers=1)
        self.assertGreater(total, 0)
        self.assertGreater(routed, 0)
        self.assertGreater(shared, 0)
        work = backend.native_moe.work_model.estimate({rank: 4 for rank in range(8)}, {rank: 0 for rank in range(8)})
        self.assertEqual(work.expert_us, 0)
        self.assertEqual(work.expert_selections, 0)
        self.assertEqual(work.expert_mma_rows, 0)
        self.assertEqual(work.expert_flops, 0)
        self.assertEqual(work.issued_weight_bytes, 0)
        self.assertGreater(work.input_us, 0)
        self.assertGreater(work.combine_us, 0)

    def test_kernel_block_choice_uses_full_local_input(self):
        self.assertEqual([block_m_for_input(n, 8, 384, 6) for n in (0, 64, 128, 4096)], [16, 16, 32, 192])
        backend = self.backend()
        full = {rank: 4096 if rank == 0 else 64 for rank in range(8)}
        valid = dict(full)
        work = backend.native_moe.work_model.estimate(full, valid)
        self.assertEqual(work.block_ms, (192, 16, 16, 16, 16, 16, 16, 16))
        self.assertEqual(work.global_valid_rows, 4544)
        self.assertEqual(work.expert_selections, 4544 * 6)
        self.assertGreater(work.dispatch_us, 0)
        uniform = {rank: 568 for rank in range(8)}
        mean_work = backend.native_moe.work_model.estimate(uniform, uniform)
        self.assertNotEqual(work.latency_us, mean_work.latency_us)

    def test_work_spectrum_preserves_selection_sum_and_unique_topk_bound(self):
        backend = self.backend()
        for rows in (0, 1, 8, 64, 512, 4544, 32768):
            with self.subTest(rows=rows):
                counts = backend.native_moe.work_model.expected_counts(rows)
                self.assertAlmostEqual(sum(counts) * 8, rows * 6, places=6)
                self.assertTrue(all(0 <= value <= rows for value in counts))

    def test_dynamic_mma_tails_keep_math_rows_when_block_partition_changes(self):
        model = self.backend().native_moe.work_model
        # Five17-row experts issue32 MMA rows each; the11-row expert issues16.
        # Each EP rank receives96 selections, matching128 valid rows *6 /8.
        received = (17.0,) * 5 + (11.0,) + (0.0,) * 42
        cases = []
        with patch.object(model, "expected_counts", return_value=received):
            for full in (64, 256, 4096):
                work = model.estimate({rank: full for rank in range(8)},
                                      {rank: 16 for rank in range(8)})
                self.assertEqual(work.expert_mma_rows, 176)
                self.assertEqual(work.expert_flops, 176 * 6 * 7168 * 3072)
                cases.append(work)
        self.assertEqual([case.block_ms[0] for case in cases], [16, 64, 192])
        self.assertEqual([case.max_expert_tiles for case in cases], [11, 6, 6])
        weight_bytes = 3 * 7168 * 3072 * (0.5 + 1 / 32)
        self.assertEqual([case.issued_weight_bytes for case in cases],
                         [11 * weight_bytes, 6 * weight_bytes, 6 * weight_bytes])

    def test_compute_bound_expert_cost_uses_dynamic_tail_math(self):
        model = self.backend().native_moe.work_model
        # Remove the bandwidth bound to expose arithmetic independently.
        model.bandwidth = float("inf")
        received = (17.0,) * 5 + (11.0,) + (0.0,) * 42
        with patch.object(model, "expected_counts", return_value=received):
            work = model.estimate({rank: 4096 for rank in range(8)},
                                  {rank: 16 for rank in range(8)})
        expected = 176 * 6 * 7168 * 3072 / model.compute * 1e6
        self.assertAlmostEqual(work.expert_us, expected)
        self.assertLess(work.expert_mma_rows, work.max_expert_tiles * 192)

    def test_subtoken_expected_occupancy_keeps_fractional_mma_work(self):
        model = self.backend().native_moe.work_model
        received = (0.25,) * 24 + (0.0,) * 24
        with patch.object(model, "expected_counts", return_value=received):
            work = model.estimate({rank: 256 for rank in range(8)},
                                  {rank: 1 for rank in range(8)})
        self.assertEqual(work.expert_mma_rows, 96)
        self.assertEqual(work.max_expert_tiles, 6)
        self.assertEqual(work.expert_selections, 48)

    def test_native_mixed_input_peak_is_independent_of_fp4_storage_peak(self):
        for sku in ("b200", "b300"):
            with self.subTest(sku=sku):
                device = native_moe.NativeMegaMoETest.context.device(sku)
                model = self.backend(device=device).native_moe.work_model
                self.assertEqual(model.compute, device.peak_compute_for("fp8_block").value)
                self.assertNotEqual(model.compute, device.peak_compute_for("mxfp4").value)
                description = model.describe()
                self.assertIn("mxf8f6f4", description["mma_opcode"])
                self.assertIn("unmeasured", description["compute_rate_grade"])
                self.assertFalse(description["fitted_parameters"])

    def test_export_time_ledger_and_profile_reset(self):
        backend = self.backend()
        backend.set_moe_step(MoEPass({rank: RankShape(64, 4, False, 16, 16, 4) for rank in range(8)}))
        backend.native_moe.estimate(64)
        result = get_latency_stats(SimpleNamespace(workers=[SimpleNamespace(latency_backend=backend, device=backend.device)]))
        ledger = result["native_megamoe_composition"]
        self.assertAlmostEqual(ledger["composed_routed_seconds"], result["native_megamoe_routed_seconds"])
        self.assertAlmostEqual(ledger["uniform_anchor_seconds"] - ledger["anchor_difference_seconds"], ledger["composed_routed_seconds"])
        self.assertEqual(result["native_megamoe_counts"]["padding_rows"], 48)
        self.assertEqual(result["native_megamoe_counts"]["routed_valid_rows"], 16)
        flops_per_mma_row = 6 * 7168 * 3072
        for prefix in ("source", "target"):
            self.assertAlmostEqual(ledger[f"{prefix}_expert_flops"] / flops_per_mma_row,
                                   ledger[f"{prefix}_mma_layer_rows"])
            self.assertGreater(ledger[f"{prefix}_issued_weight_bytes"], 0)
        backend.native_moe.reset_stats()
        self.assertEqual(backend.native_moe.composition, {})
        self.assertEqual(backend.native_moe.counts, {})
