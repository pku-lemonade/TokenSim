from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.model_config import ModelCatalog, ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.config.weight_memory import deepseek_v4_weight_memory
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.moe.placement import build_expert_placement


class DeepSeekV4ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = yaml.safe_load(Path("data/models/deepseek-v4-pro.yaml").read_text())
        cls.model = ModelSpec.from_mapping(cls.raw)
        cls.hf = json.loads(
            Path("data/reference/agentx-model/deepseek-v4-pro.hf.json").read_text()
        )

    def test_all_architecture_dimensions_match_saved_hf_config(self):
        m, hf = self.model, self.hf
        for field, key in (
            ("hidden_size", "hidden_size"),
            ("num_layers", "num_hidden_layers"),
            ("num_attention_heads", "num_attention_heads"),
            ("num_key_value_heads", "num_key_value_heads"),
            ("head_dim", "head_dim"),
            ("vocab_size", "vocab_size"),
            ("max_position_embeddings", "max_position_embeddings"),
            ("sliding_window", "sliding_window"),
        ):
            self.assertEqual(getattr(m, field), hf[key], field)
        spec = m.deepseek_v4
        for field in (
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "qk_rope_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "hc_mult",
            "hc_sinkhorn_iters",
            "num_hash_layers",
            "num_nextn_predict_layers",
            "routed_scaling_factor",
            "scoring_func",
            "topk_method",
        ):
            self.assertEqual(getattr(spec, field), hf[field], field)
        self.assertEqual(spec.compress_ratios, tuple(hf["compress_ratios"][:61]) + (1,))
        self.assertEqual(m.moe.num_experts, hf["n_routed_experts"])
        self.assertEqual(m.moe.num_experts_per_tok, hf["num_experts_per_tok"])
        self.assertEqual(m.moe.num_shared_experts, hf["n_shared_experts"])
        self.assertEqual(m.moe_intermediate_size, hf["moe_intermediate_size"])

    def test_first_three_layers_are_hash_moe_not_dense(self):
        layers = self.model.layer_specs(include_mtp=False)
        self.assertEqual(len(layers), 61)
        self.assertEqual(
            Counter(l.attention_kind for l in layers), {"csa": 30, "hca": 31}
        )
        self.assertEqual(
            [l.layer_id for l in layers if l.moe_routing == "hash"], [0, 1, 2]
        )
        self.assertEqual(self.model.moe_layer_indices(), list(range(61)))
        self.assertEqual(self.model.dense_layer_count(), 0)
        mtp = self.model.layer_specs()[-1]
        self.assertTrue(mtp.is_mtp)
        self.assertEqual(mtp.attention_kind, "swa")
        self.assertEqual(mtp.compress_ratio, 1)
        self.assertEqual(mtp.moe_routing, "learned")

    def test_parameter_and_activation_dtypes_remain_separate(self):
        m = self.model
        self.assertEqual(m.operator_dtype("mhc"), "fp32")
        self.assertEqual(m.operator_dtype("mhc_residual"), "bf16")
        self.assertEqual(m.activation_dtype, "bf16")
        self.assertEqual(m.kv_cache_dtype, "fp8")
        self.assertEqual(m.operator_dtype("routed_expert"), "mxfp4")
        self.assertEqual(m.operator_dtype("attention_gemm"), "fp8_block")
        self.assertEqual(m.operator_dtype("shared_expert"), "fp8_block")
        self.assertEqual(m.operator_dtype("indexer_weights_proj"), "bf16")

    def test_incomplete_or_inconsistent_native_structure_is_rejected(self):
        for mutate in (
            lambda r: r["deepseek_v4"]["compress_ratios"].pop(),
            lambda r: r["deepseek_v4"]["compress_ratios"].__setitem__(-1, 4),
            lambda r: r["deepseek_v4"].__setitem__("num_hash_layers", 62),
            lambda r: r["operator_dtypes"].pop("mhc_residual"),
            lambda r: r.__setitem__("kv_cache_dim", 576),
            lambda r: r["moe"].update(num_moe_layers=58, first_k_dense_replace=3),
            lambda r: r.__setitem__("activation", "gelu"),
        ):
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.assertRaises(ConfigurationError):
                ModelSpec.from_mapping(raw)

    def test_psla_cannot_silently_restore_proxy_geometry(self):
        moe = PSLAConfig.from_file("data/psla/deepseek-v4-pro.json").moe_config
        self.assertEqual(
            self.model.with_moe_override(moe).deepseek_v4, self.model.deepseek_v4
        )
        for field, value in (
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
            ("moe_intermediate_size", 2048),
            ("num_key_value_heads", 128),
        ):
            invalid = replace(moe, **{field: value})
            with self.assertRaises(ConfigurationError):
                self.model.with_moe_override(invalid)

    def test_model_serialization_keeps_layer_and_dtype_metadata(self):
        data = self.model.to_dict()
        self.assertEqual(data["architecture"], "deepseek_v4")
        self.assertEqual(len(data["layers"]), 62)
        self.assertTrue(data["layers"][-1]["is_mtp"])
        self.assertEqual(data["deepseek_v4"]["hc_mult"], 4)
        self.assertEqual(data["sliding_window"], 128)
        self.assertEqual(data["max_position_embeddings"], 1048576)
        self.assertEqual(data["operator_dtypes"]["routed_expert"], "mxfp4")
        self.assertGreater(data["total_params"], 1.5e12)
        self.assertLess(data["total_params"], 1.7e12)


class DeepSeekV4WeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")
        cls.parallel = ClusterConfig.from_file(
            "data/clusters/8_b200/dep8.json"
        ).effective_parallel_config()

    def test_dep8_owns_48_experts_per_rank_and_meets_weight_log_tolerance(self):
        for sku, logged in (("b200", 125.88), ("b300", 126.08)):
            for dp in range(8):
                report = deepseek_v4_weight_memory(
                    self.model, self.parallel, ParallelRankInfo(dp_rank=dp)
                )
                self.assertEqual(report["local_experts"], 48)
                self.assertEqual(report["moe_tensor_parallel_size"], 1)
                self.assertLess(
                    abs(report["total_bytes"] / 2**30 / logged - 1), 0.02, (sku, dp)
                )

    def test_tensor_shapes_follow_low_rank_grouped_attention(self):
        report = deepseek_v4_weight_memory(self.model, self.parallel)
        tensors = {t["name"]: t for t in report["tensors"]}
        self.assertEqual(tensors["layers.0.attn.fused_wqa_wkv"]["shape"], [2048, 7168])
        self.assertEqual(tensors["layers.0.attn.wq_b"]["shape"], [65536, 1536])
        self.assertEqual(tensors["layers.0.attn.wo_a"]["shape"], [16384, 4096])
        self.assertEqual(tensors["layers.0.attn.wo_b"]["shape"], [7168, 16384])
        self.assertEqual(tensors["layers.2.indexer.wq_b"]["shape"], [8192, 1536])
        self.assertNotIn("layers.0.indexer.wq_b", tensors)
        self.assertNotIn("mtp.61.attn.compressor.fused_wkv_wgate", tensors)

    def test_quantization_scales_and_nonquantized_tensors_are_counted(self):
        report = deepseek_v4_weight_memory(self.model, self.parallel)
        tensors = {t["name"]: t for t in report["tensors"]}
        t = tensors["layers.0.experts.gate_up"]
        self.assertEqual(t["shape"], [48, 6144, 7168])
        self.assertEqual(t["weight_bytes"], 48 * 6144 * 7168 // 2)
        self.assertEqual(t["scale_bytes"], 48 * 6144 * (7168 // 32))
        self.assertEqual(tensors["layers.0.attn.wq_b"]["scale_bytes"], 512 * 12)
        self.assertEqual(tensors["embedding"]["dtype"], "bf16")
        self.assertEqual(tensors["embedding"]["scale_bytes"], 0)
        self.assertEqual(
            tensors["layers.0.ffn.tid2eid"]["metadata_bytes"], 129280 * 6 * 8
        )
        self.assertEqual(tensors["layers.0.hc_attn.fn"]["weight_bytes"], 24 * 28672 * 4)
        self.assertEqual(tensors["layers.0.attn.compressor.ape"]["dtype"], "fp32")
        self.assertEqual(
            report["total_bytes"],
            report["weight_bytes"] + report["scale_bytes"] + report["metadata_bytes"],
        )
        self.assertEqual(
            sum(
                t["weight_bytes"] + t["scale_bytes"] + t["metadata_bytes"]
                for t in report["tensors"]
            ),
            report["total_bytes"],
        )

    def test_mtp_reuses_target_embeddings_and_head_and_adds_one_module(self):
        with_mtp = deepseek_v4_weight_memory(self.model, self.parallel)
        without_mtp = deepseek_v4_weight_memory(
            self.model, self.parallel, include_mtp=False
        )
        self.assertGreater(with_mtp["total_bytes"], without_mtp["total_bytes"])
        self.assertEqual(
            with_mtp["components"]["embedding"], without_mtp["components"]["embedding"]
        )
        self.assertEqual(
            with_mtp["components"]["lm_head"], without_mtp["components"]["lm_head"]
        )
        self.assertEqual(
            with_mtp["components"]["routed_expert"]["weight_bytes"] * 61,
            without_mtp["components"]["routed_expert"]["weight_bytes"] * 62,
        )
        self.assertEqual(
            with_mtp["components"]["mtp_input_projection"]["tensor_count"], 2
        )

    def test_wide_ep_does_not_divide_experts_again_by_tp(self):
        ep = ParallelConfig(
            tensor_parallel_size=4,
            data_parallel_size=2,
            enable_expert_parallel=True,
            expert_parallel_size=8,
        )
        report = deepseek_v4_weight_memory(self.model, ep)
        reference = deepseek_v4_weight_memory(self.model, self.parallel)
        self.assertEqual(report["local_experts"], 48)
        self.assertEqual(
            report["components"]["routed_expert"],
            reference["components"]["routed_expert"],
        )
        self.assertTrue(report["shared_experts_replicated_across_tp"])
        self.assertEqual(
            report["components"]["shared_expert"],
            reference["components"]["shared_expert"],
        )
        self.assertLess(
            report["components"]["attention"]["weight_bytes"],
            reference["components"]["attention"]["weight_bytes"],
        )

    def test_tp_without_ep_shards_intermediate_width_instead_of_expert_ids(self):
        parallel = ParallelConfig(tensor_parallel_size=8)
        report = deepseek_v4_weight_memory(self.model, parallel)
        tensors = {t["name"]: t for t in report["tensors"]}
        self.assertEqual(report["local_experts"], 384)
        self.assertEqual(report["moe_tensor_parallel_size"], 8)
        self.assertEqual(tensors["layers.0.experts.gate_up"]["shape"], [384, 768, 7168])
        self.assertEqual(tensors["layers.0.experts.down"]["shape"], [384, 7168, 384])
        self.assertEqual(tensors["layers.0.shared.gate_up"]["shape"], [768, 7168])

    def test_fused_ep_layout_keeps_hash_table_int32_and_respects_sequence_parallel(
        self,
    ):
        parallel = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
        fused = deepseek_v4_weight_memory(self.model, parallel, layout_backend="fused")
        mega = deepseek_v4_weight_memory(
            self.model, parallel, layout_backend="mega_moe"
        )
        self.assertFalse(fused["shared_experts_replicated_across_tp"])
        self.assertTrue(mega["shared_experts_replicated_across_tp"])
        self.assertEqual(
            fused["components"]["hash_routing"]["metadata_bytes"] * 2,
            mega["components"]["hash_routing"]["metadata_bytes"],
        )
        with self.assertRaises(ConfigurationError):
            deepseek_v4_weight_memory(
                self.model, ParallelConfig(), layout_backend="mega_moe"
            )

    def test_cache_budget_uses_tensor_bytes_instead_of_global_dtype(self):
        for sku in ("b200", "b300"):
            device = self.hardware.device(sku)
            placement = build_expert_placement(
                self.model.moe, self.parallel, self.model.num_layers
            )
            cache = CacheConfig(
                256, device, self.model, self.parallel, ParallelRankInfo(), placement
            )
            report = deepseek_v4_weight_memory(self.model, self.parallel)
            self.assertEqual(cache.model_param_size, report["total_bytes"])
            self.assertNotEqual(cache.model_param_size, self.model.total_params() / 8)
            self.assertGreater(cache.num_gpu_blocks, 0)
            self.assertEqual(cache.to_dict()["weight_memory"]["local_experts"], 48)

    def test_unsupported_pp_or_rank_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            deepseek_v4_weight_memory(
                self.model, ParallelConfig(pipeline_parallel_size=2)
            )
        for dp_rank in (-1, 8):
            with self.assertRaises(ConfigurationError):
                deepseek_v4_weight_memory(
                    self.model, self.parallel, ParallelRankInfo(dp_rank=dp_rank)
                )

    def test_native_capacity_cannot_be_overridden_from_reference_logs(self):
        with self.assertRaises(ConfigurationError):
            CacheConfig(
                256,
                self.hardware.device("b200"),
                self.model,
                self.parallel,
                kv_cache_capacity_tokens_per_dp_rank=2742953,
            )

    def test_complete_clusters_are_one_node_eight_gpus_without_capacity_overrides(self):
        for sku in ("b200", "b300"):
            cluster = ClusterConfig.from_file(f"data/clusters/8_{sku}/dep8.json")
            parallel = cluster.effective_parallel_config()
            workers = cluster.workers(parallel)
            self.assertEqual(len(workers), 8)
            self.assertEqual(parallel.world_size, 8)
            self.assertEqual(parallel.tensor_parallel_size, 1)
            self.assertEqual(parallel.data_parallel_size, 8)
            self.assertEqual(parallel.expert_parallel_group_size, 8)
            self.assertEqual(parallel.expert_parallel_scope, "global")
            self.assertEqual(parallel.data_parallel_size_local, 8)
            self.assertEqual({w.network for w in workers}, {"node0"})
            self.assertIsNone(cluster.kv_cache_capacity_tokens_total)
            self.assertIsNone(cluster.kv_cache_capacity_tokens_per_dp_rank)

    def test_matrix_model_and_cluster_paths_resolve_to_native_dep8(self):
        from util.agentx_reference import load_experiments

        for exp in load_experiments(Path("data/experiments/agentx.yaml")):
            model = PSLAConfig.from_file(exp["model_config_path"])
            self.assertEqual(model.model, "DeepSeek-V4-Pro")
            cluster = ClusterConfig.from_file(exp["cluster"])
            parallel = cluster.effective_parallel_config()
            self.assertEqual(parallel.world_size, exp["num_gpus"])
            self.assertEqual(parallel.tensor_parallel_size, exp["parallelism"]["tp"])
            self.assertEqual(parallel.data_parallel_size, exp["parallelism"]["dp"])
            self.assertEqual(
                parallel.expert_parallel_group_size, exp["parallelism"]["ep"]
            )


if __name__ == "__main__":
    unittest.main()
