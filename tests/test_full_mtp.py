"""DSV4 complete draft-pass shape and time-accounting regressions."""

from unittest import TestCase
from unittest.mock import Mock, patch
from types import SimpleNamespace

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.latency.deepseek_v4_moe import MoEPass, RankShape
from TokenSim.llm.llm_request import Request
from util.results import get_latency_stats
from TokenSim.config.psla_config import LLMResult


class FullMTPTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = HardwareContext.load("data")
        cls.model = cls.context.model("DeepSeek-V4-Pro")

    def backend(self, *, tp=1, dp=8, enabled=True, communicator=None):
        device = self.context.device("b200")
        parallel = ParallelConfig(tensor_parallel_size=tp, data_parallel_size=dp,
                                  enable_expert_parallel=True)
        return OperatorTableLatencyBackend(
            device=device, model=self.model, parallel_config=parallel,
            rank_info=ParallelRankInfo.from_global_rank(0, parallel),
            package=self.context.operator_package(device.device_id, "vllm"),
            attention_backend="native_dsv4", mhc_backend="native_dsv4",
            moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
            expected_decode_query_len=4 if enabled else 1, communicator=communicator,
        )

    def decode_requests(self, count=16, *, width=4):
        result = []
        for i in range(count):
            req = Request(i, 8192, 20, 256)
            req.generation_idx = 1
            req.num_computed_tokens = 8192
            req.scheduled_tokens = 2
            req.scheduled_query_tokens = width
            result.append(req)
        return result

    def set_uniform_step(self, backend, *, tokens=64, count=16, context=False):
        step = MoEPass({rank: RankShape(tokens, None if context else 4, context, count)
                       for rank in range(backend.parallel_config.data_parallel_size)})
        backend.set_moe_step(step)

    def test_three_passes_cover_full_verify_and_two_request_batches(self):
        backend = self.backend()
        self.set_uniform_step(backend)
        costs = backend.native_mtp.estimate(self.decode_requests())
        c = backend.native_mtp.counts
        self.assertEqual(c["passes"], 3)
        self.assertEqual(c["initial_passes"], 1)
        self.assertEqual(c["decode_passes"], 2)
        self.assertEqual(c["forward_rows"], 96)
        self.assertEqual(c["e_proj_rows"], 96)
        self.assertEqual(c["h_proj_rows"], 384)
        self.assertEqual(c["logits_rows"], 48)
        self.assertTrue(all(costs[name] > 0 for name in ("gemm", "attention", "moe", "elementwise", "lm_head")))
        self.assertEqual(backend.native_mtp.initial_moe.counts["rank_estimates"], 1)
        self.assertEqual(backend.native_mtp.decode_moe.counts["rank_estimates"], 2)

    def test_partial_prefill_executes_full_initial_draft(self):
        backend = self.backend()
        req = Request(0, 8192, 1, 256)  # warmup/partial prefill still invokes the drafter
        req.scheduled_tokens = req.scheduled_query_tokens = 512
        self.set_uniform_step(backend, tokens=512, count=1, context=True)
        backend.native_mtp.estimate([req])
        counts = backend.native_mtp.counts
        self.assertEqual(counts["forward_rows"], 514)
        self.assertEqual(counts["h_proj_rows"], 2056)
        self.assertEqual(counts["logits_rows"], 3)
        self.assertEqual(counts["eager_passes"], 1)
        self.assertEqual(counts["graph_passes"], 2)

    def test_tp_sp_padding_and_draft_collectives_are_explicit(self):
        comm = Mock()
        comm.describe.return_value = {"ep_group_link": "node", "ep_group_size": 8}
        comm.estimate_tp_collective.return_value = 1e-6
        backend = self.backend(tp=4, dp=2, communicator=comm)
        backend.set_moe_step(MoEPass({0: RankShape(16, 4, False, 4),
                                     1: RankShape(32, 4, False, 8)}))
        costs = backend.native_mtp.estimate(self.decode_requests(4))
        c = backend.native_mtp.counts
        self.assertEqual(c["forward_rows"], 48)  # target-like 32, then 8 + 8
        self.assertEqual(c["e_proj_rows"], 12)   # sequence-parallel shards
        self.assertEqual(c["h_proj_rows"], 48)
        self.assertEqual(c["logits_rows"], 24)   # captured routine includes padded head rows
        self.assertEqual(comm.estimate_tp_collective.call_count, 9)
        self.assertAlmostEqual(costs["comm"], 9)
        comm.estimate_ep_all2all.assert_not_called()

    def test_swa_has_no_indexer_and_ignores_old_context_beyond_window(self):
        backend = self.backend()
        short = self.decode_requests()
        long = self.decode_requests()
        for req in long:
            req.num_computed_tokens = 1048000
        self.assertEqual(backend.native_mtp._swa(short, 0), backend.native_mtp._swa(long, 0))
        self.assertIn("SWA only", backend.native_mtp.describe()["attention"])
        self.assertNotIn("csa", backend.native_mtp.describe()["attention"])

    def test_draft_component_conservation_and_profile_reset(self):
        backend = self.backend()
        self.set_uniform_step(backend)
        latency = backend.estimate_step_latency(self.decode_requests())
        self.assertAlmostEqual(sum(backend.stats.component_seconds.values()), latency)
        mtp = backend.native_mtp
        for key, value in mtp.component_seconds.items():
            self.assertLessEqual(value, backend.stats.component_seconds[key])
        self.assertAlmostEqual(backend.stats.component_seconds["attention"],
                               backend.native_attention.seconds + mtp.component_seconds["attention"])
        result = get_latency_stats(SimpleNamespace(workers=[SimpleNamespace(latency_backend=backend, device=backend.device)]))
        self.assertEqual(result["native_dsv4_mtp_counts"]["passes"], 3)
        self.assertAlmostEqual(result["native_megamoe_routed_seconds"] + result["native_megamoe_shared_seconds"],
                               backend.stats.component_seconds["moe"])
        self.assertTrue({"native_megamoe_routed_seconds", "native_megamoe_shared_seconds",
                         "native_megamoe_counts", "native_megamoe_max_rank_rows"} <= LLMResult.__dataclass_fields__.keys())
        backend.reset_profile_stats()
        self.assertEqual(mtp.counts, {})
        self.assertEqual(mtp.component_seconds, {})

    def test_disabled_mtp_has_no_draft_estimator(self):
        self.assertIsNone(self.backend(enabled=False).native_mtp)
