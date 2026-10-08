"""Source allocation oracles and alias/lifetime regressions for native DSV4."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.deepseek_v4_buffers import MIB, PHASES, live_phases, mega_moe_storage
from TokenSim.config.memory_budget import deepseek_v4_memory_budget
from TokenSim.config.moe_execution import MoEExecutionConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext


class NativeMemoryBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")
        cls.parallel = ParallelConfig(data_parallel_size=8, data_parallel_size_local=8,
                                      enable_expert_parallel=True)

    def budget(self, *, tokens=8192, seqs=49, spec=3, parallel=None, sku="b200", execution=None):
        return deepseek_v4_memory_budget(
            self.model, self.hardware.device(sku), 133615585400, parallel or self.parallel,
            utilization=.9, max_num_batched_tokens=tokens, max_num_seqs=seqs,
            speculative_tokens=spec, moe_execution=execution,
        )

    def test_all_preimplementation_native_allocation_oracles(self):
        # Saved host-ABI/source interval derivations predate the runtime port.
        fixture = json.loads(Path("data/reference/agentx-capture-rows/buffer-geometry-probe.json").read_text())
        for row in fixture["cases"]:
            parallel = ParallelConfig(tensor_parallel_size=row["expert_parallel_size"],
                                      enable_expert_parallel=True)
            with self.subTest(tokens=row["unsharded_scheduler_tokens"], ep=row["expert_parallel_size"],
                              sms=row["SM_scenario"], shared=row["shared_experts"]):
                storage = mega_moe_storage(self.model, parallel, row["unsharded_scheduler_tokens"],
                                          shared=bool(row["shared_experts"]), sms=row["SM_scenario"])
                self.assertEqual(storage["bytes"], row["total_bytes"])
                self.assertEqual(storage["ring_tokens"], row["ring_tokens"])
                self.assertEqual(storage["components"], {key: value for key, value in row["components_bytes"].items()
                                                       if key != "shared_l1_x_alias"})
                self.assertEqual(storage["aliases"], {"shared_l1_x": "input_x"})
                self.assertEqual(storage["intervals"][-1]["end_byte"], storage["bytes"])
                self.assertEqual(storage["cache_instances_per_rank"], 1)

    def test_persistent_capacity_is_unsharded_and_buffers_are_shared_across_layers(self):
        dep = mega_moe_storage(self.model, self.parallel, 8192)
        tp4 = ParallelConfig(tensor_parallel_size=4, data_parallel_size=2,
                             data_parallel_size_local=2, enable_expert_parallel=True)
        sharded = mega_moe_storage(self.model, tp4, 8192)
        self.assertEqual(dep["bytes"], 2246689664)
        self.assertEqual(sharded["bytes"], dep["bytes"])
        self.assertEqual(sharded["aligned_tokens"], 9600)
        self.assertEqual(dep["cache_key_geometry"], sharded["cache_key_geometry"])

    def test_steady_layer_rhs_retains_three_distinct_hyper_residuals(self):
        plan = live_phases(self.model, self.parallel, 196, 49, kind="csa")
        phase = PHASES.index("mhc_ffn")
        residuals = [row for row in plan["storages"] if row["storage"] in
                     ("caller_residual", "attention_residual", "ffn_residual")
                     and row["begin_phase"] <= phase < row["end_phase"]]
        self.assertEqual(len(residuals), 3)
        self.assertEqual(sum(row["bytes"] for row in residuals), 3 * 196 * 4 * 7168 * 2)

    def test_sp_gather_views_keep_tail_storage_and_tp1_output_is_fp8(self):
        parallel = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
        sharded = live_phases(self.model, parallel, 20, 49, kind="csa")
        allocations = {row["storage"]: row["bytes"] for row in sharded["storages"]}
        self.assertEqual((sharded["sp_rows"], sharded["gather_owner_rows"]), (3, 24))
        self.assertEqual(allocations["attention_input"], 24 * 7168 * 2)
        self.assertEqual(allocations["reduce_scatter_padding"], 24 * 7168 * 2)
        full = live_phases(self.model, self.parallel, 20, 49, kind="csa")
        allocations = {row["storage"]: row["bytes"] for row in full["storages"]}
        self.assertEqual(allocations["attention_output"], 20 * 128 * 512)
        self.assertNotIn("output_quantized_values", allocations)

    def test_draft_sp_views_retain_full_normalized_hidden(self):
        parallel = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
        plan = live_phases(self.model, parallel, 20, 49, kind="swa", draft=True)
        allocations = {row["storage"]: row["bytes"] for row in plan["storages"]}
        self.assertEqual(allocations["mtp_normalized_hidden"], 20 * 4 * 7168 * 2)
        self.assertEqual(allocations["attention_residual"], 3 * 4 * 7168 * 2)
        self.assertNotIn("indexer_logits", allocations)

    def test_context_loop_rhs_and_decode_logits_have_distinct_owners(self):
        eager = live_phases(self.model, self.parallel, 8192, 49, kind="csa")
        allocations = {row["storage"]: row["bytes"] for row in eager["storages"]}
        self.assertEqual(allocations["indexer_logits"], 512 * MIB)
        self.assertEqual(allocations["previous_indexer_logits"], 512 * MIB)
        capture = live_phases(self.model, self.parallel, 196, 49, kind="csa", graph=True)
        allocations = {row["storage"]: row["bytes"] for row in capture["storages"]}
        self.assertEqual(allocations["indexer_logits"], 196 * (1048576 // 4) * 4)
        self.assertNotIn("previous_indexer_logits", allocations)

    def test_named_persistent_allocations_and_workspace_alias_are_not_layer_copies(self):
        budget = self.budget()
        persistent = budget["persistent_components"]
        self.assertEqual(persistent["target_mtp_hidden"], 8192 * 4 * 7168 * 2)
        self.assertEqual(persistent["drafter_hidden"], persistent["target_mtp_hidden"])
        self.assertEqual(persistent["target_topk"], persistent["draft_topk"])
        self.assertEqual(persistent["indexer_shared_workspace"], 714080256)
        self.assertEqual(persistent["compressed_rope_cache"], 268435456)
        self.assertEqual(persistent["swa_rope_cache"], 268435456)
        self.assertEqual(budget["deductions"]["native_persistent_buffers"], sum(persistent.values()))

    def test_graph_pool_tracks_descriptors_and_lifetimes_not_model_layers(self):
        graph = self.budget()["graph_storage"]
        self.assertEqual((graph["target_descriptors"], graph["draft_prefill_descriptors"],
                          graph["draft_decode_descriptors"]), (49, 49, 12))
        self.assertEqual(graph["capture_rows"], 196)
        self.assertEqual(graph["shared_pool_live_peak_bytes"], max(row["peak_bytes"] for row in graph["plans"]))
        self.assertEqual(graph["bytes"], graph["shared_pool_live_peak_bytes"]
                         + graph["persistent_target_output_bytes"] + graph["driver_increment_proxy_bytes"])
        self.assertIn("unmeasured", graph["driver_grade"])

    def test_mtp_iterations_reuse_persistent_buffers_and_disabled_module_allocates_none(self):
        one, three, disabled = self.budget(spec=1), self.budget(spec=3), self.budget(spec=0)
        self.assertEqual(one["persistent_components"], three["persistent_components"])
        for key in ("target_mtp_hidden", "drafter_hidden", "draft_topk", "swa_rope_cache"):
            self.assertEqual(disabled["persistent_components"][key], 0)
        self.assertEqual(disabled["graph_storage"]["draft_prefill_descriptors"], 0)
        self.assertFalse(any(row["draft"] for row in disabled["activation_live_plans"]))

    def test_generic_moe_cannot_claim_a_native_ring_and_unfused_shared_is_transient(self):
        generic = self.budget(execution=MoEExecutionConfig())
        self.assertIsNone(generic["native_mega_storage"])
        self.assertIn("generic_ep_communication_envelope", generic["persistent_components"])
        unfused = self.budget(execution=MoEExecutionConfig(backend="deep_gemm_amxf4_mega_moe",
                                                         shared_expert_fusion=False))
        self.assertEqual(unfused["native_mega_storage"]["components"]["shared_l2_x"], 0)
        self.assertTrue(any(row["storage"] == "unfused_shared_activations"
                            for plan in unfused["activation_live_plans"] for row in plan["storages"]))

    def test_capacity_is_derived_from_full_positive_deductions_with_reserve_retained(self):
        cache = CacheConfig(256, self.hardware.device("b200"), self.model, self.parallel,
                            usable_memory_fraction=.9)
        budget = cache.memory_budget
        self.assertEqual(budget["deductions"]["runtime_and_nccl_reserve"], 4169138176)
        self.assertEqual(budget["available_kv_bytes"], budget["hardware_budget_bytes"] - sum(budget["deductions"].values()))
        self.assertEqual(sum(budget["activation_components"].values()), budget["deductions"]["peak_activation_estimate"])
        self.assertEqual(cache.num_gpu_blocks, budget["available_kv_bytes"] // cache.physical_slot_bytes)
        self.assertGreaterEqual(budget["available_kv_bytes"] - cache.num_gpu_blocks * cache.physical_slot_bytes, 0)
        self.assertLess(budget["available_kv_bytes"] - cache.num_gpu_blocks * cache.physical_slot_bytes, cache.physical_slot_bytes)
        with self.assertRaises(ConfigurationError):
            CacheConfig(256, self.hardware.device("b200"), self.model, self.parallel,
                        kv_cache_capacity_tokens_per_dp_rank=2742953)

    def test_invalid_geometry_and_budget_exhaustion_are_explicit(self):
        with self.assertRaises(ConfigurationError):
            mega_moe_storage(self.model, ParallelConfig(), 8192)
        for tokens, seqs, speculative in ((0, 49, 3), (1.5, 49, 3), (8192, 0, 3), (8192, 49, -1)):
            with self.subTest(tokens=tokens, seqs=seqs), self.assertRaises(ConfigurationError):
                self.budget(tokens=tokens, seqs=seqs, spec=speculative)
        with self.assertRaises(ConfigurationError):
            deepseek_v4_memory_budget(self.model, self.hardware.device("b200"), 200 * 2**30,
                self.parallel, utilization=.9, max_num_batched_tokens=8192, max_num_seqs=49)
