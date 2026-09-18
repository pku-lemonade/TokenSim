import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.run_deepseek_mla import (
    ROOT, ShapeDiscovery, configuration, generate_package, simulate,
)
from TokenSim.af import AFComputeQuery
from TokenSim.af.runtime import kv_head_partitions
from TokenSim.operator_data import (
    Calibration, EvidenceGrade, HardwareProfile, MissingOperatorDataError,
    ModelOperatorManifest, Observation, OperatorDataComputeLatencyProvider,
    OperatorDataLookup, SourceRecord, estimate_operator,
)
from TokenSim.operator_data.schema import validate_batch_key


ASSUMPTIONS = json.loads((ROOT / "data/operator_data/deepseek-v3-theoretical.json").read_text())
HW = HardwareProfile("a100-assumed", "analytical", {"bf16": 312e12}, 1555e9)


def core_key(**overrides):
    key = dict(device="a100-assumed", backend="analytical", activation_dtype="bf16",
        kv_cache_dtype="bf16", phase="decode", execution_mode="absorbed", batch_size=1,
        query_len=1, context_lengths="16384", num_heads=128, kv_lora_rank=512,
        rope_head_dim=64, nope_head_dim=128, v_head_dim=128)
    key.update(overrides)
    return key


def query(shape, contexts=(16, 24), phase="decode"):
    tokens = len(contexts) if phase == "decode" else sum(contexts)
    local = dict(tokens=tokens, hidden_size=shape.hidden_size, global_hidden_size=shape.hidden_size,
                 **shape.mla_dimensions.local_shape(shape.num_attention_heads, shape.tp_size))
    return AFComputeQuery(iteration_id=0, block_id=0, layer_id=0, phase=phase,
        stage="attention", token_count=tokens, context_lengths=contexts,
        model_id=shape.model_id, dtype="bf16", device_type="a100-assumed", worker_id=0,
        local_token_count=tokens, local_shape=local, layer_type="moe", original_token_count=tokens)


class MLAFormulaTest(unittest.TestCase):
    def test_decode_cache_is_one_latent_vector_not_per_head_k_and_v(self):
        result = estimate_operator("mla_attention", core_key(), HW)
        inputs = result.details["formula_inputs"]
        self.assertEqual(inputs["resident_cache_bytes"], 18 * 1024**2)
        self.assertEqual(inputs["cache_read_bytes"], 18 * 1024**2)
        self.assertEqual(inputs["cache_write_bytes"], 1152)
        self.assertEqual(inputs["matmul_flops"], 2*128*16384*(1024+64))
        self.assertEqual(result.kv_bytes, 18*1024**2+1152)
        half_heads = estimate_operator("mla_attention", core_key(num_heads=64), HW)
        self.assertEqual(half_heads.kv_bytes, result.kv_bytes)
        self.assertEqual(half_heads.flops*2, result.flops)

    def test_ragged_batch_counts_work_but_only_one_launch(self):
        c = Calibration(offset_us=3)
        batch = estimate_operator("mla_attention", core_key(batch_size=2, context_lengths="10,20"), HW, c)
        one = estimate_operator("mla_attention", core_key(context_lengths="10"), HW, c)
        two = estimate_operator("mla_attention", core_key(context_lengths="20"), HW, c)
        self.assertEqual(batch.flops, one.flops+two.flops)
        self.assertEqual(batch.kv_bytes, one.kv_bytes+two.kv_bytes)
        self.assertEqual(batch.details["fixed_overhead_us"], 3)
        self.assertAlmostEqual(batch.latency_us, max(batch.compute_time_us, batch.memory_time_us)+3)
        fp8 = estimate_operator("mla_attention", core_key(kv_cache_dtype="fp8"), HW)
        bf16 = estimate_operator("mla_attention", core_key(), HW)
        self.assertEqual(fp8.kv_bytes*2, bf16.kv_bytes)
        self.assertEqual(fp8.flops, bf16.flops)

    def test_prefill_expanded_causal_core(self):
        result = estimate_operator("mla_attention", core_key(phase="prefill", execution_mode="expanded",
            batch_size=2, query_len=8, context_lengths="8,8"), HW)
        inputs = result.details["formula_inputs"]
        self.assertEqual(inputs["matmul_flops"], 2*128*(2*36)*(128+64+128))
        self.assertEqual(inputs["cache_read_bytes"], 0)
        self.assertEqual(result.kv_bytes, 16*576*2)
        with self.assertRaisesRegex(ValueError, "cached/chunked"):
            validate_batch_key("mla_attention", core_key(phase="prefill", execution_mode="expanded",
                query_len=8, context_lengths="16"))

    def test_grouped_gemm_has_head_local_weights_and_one_overhead(self):
        key = dict(device="a100-assumed", backend="analytical", weight_dtype="bf16",
                   activation_dtype="bf16", m=2, n=512, k=128, groups=128)
        result = estimate_operator("batched_gemm", key, HW, Calibration(offset_us=3))
        self.assertEqual(result.flops, 2*2*512*128*128)
        self.assertEqual(result.weight_bytes, 512*128*128*2)
        self.assertAlmostEqual(result.latency_us, max(result.compute_time_us, result.memory_time_us)+3)

    def test_invalid_core_keys_fail_before_computation(self):
        for change in (dict(batch_size=2), dict(execution_mode="expanded"), dict(query_len=2),
                       dict(phase="unknown"), dict(kv_lora_rank=0), dict(rope_head_dim=3)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                estimate_operator("mla_attention", core_key(**change), HW)


class MLAIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = configuration()
        cls.shape = cls.config[2]

    def test_metadata_matches_official_v3_config_and_validates(self):
        shape = self.shape
        reference = json.loads((ROOT / "data/models/deepseek-v3-hf-config.json").read_text())
        for local, official in (("q_lora_rank", "q_lora_rank"), ("kv_lora_rank", "kv_lora_rank"),
                               ("rope_head_dim", "qk_rope_head_dim"), ("nope_head_dim", "qk_nope_head_dim"),
                               ("v_head_dim", "v_head_dim")):
            self.assertEqual(getattr(shape, local), reference[official])
        self.assertEqual(shape.head_dim, 192)
        self.assertEqual(shape.kv_dimension, 576)
        self.assertEqual(replace(shape, q_lora_rank=1024).kv_dimension, 576)
        for field in ("q_lora_rank", "kv_lora_rank", "rope_head_dim", "nope_head_dim", "v_head_dim"):
            for value in (0, -1, 100000, True):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    ModelOperatorManifest({shape.model_id: replace(shape, **{field: value})})

    def test_provider_decomposes_actual_mla_projection_dimensions(self):
        discovery = ShapeDiscovery(ASSUMPTIONS)
        provider = OperatorDataComputeLatencyProvider(discovery,
            ModelOperatorManifest({self.shape.model_id: self.shape}), "analytical")
        result = provider.estimate(query(self.shape))
        components = {c["name"]: c for c in result.details["components"]}
        self.assertNotIn("qkv_projection", components)
        self.assertEqual(components["mla_q_up"]["key"]["n"], 128*192)
        self.assertEqual(components["mla_q_absorb"]["key"]["groups"], 128)
        self.assertEqual(components["mla_q_absorb"]["key"]["n"], 512)
        self.assertEqual(components["output_projection"]["key"]["k"], 128*128)
        self.assertEqual(components["mla_attention_core"]["key"]["context_lengths"], "16,24")
        self.assertNotIn("q_lora_rank", components["mla_attention_core"]["key"])
        tables = {s["table"] for s in discovery.shapes.values()}
        self.assertNotIn("generation_attention", tables)
        self.assertNotIn("generation_attention_batch", tables)
        prefill = provider.estimate(query(self.shape, (8, 8), "prefill"))
        names = {c["name"] for c in prefill.details["components"]}
        self.assertIn("mla_kv_expand", names)
        self.assertNotIn("mla_q_absorb", names)
        self.assertNotIn("mla_value_expand", names)
        with self.assertRaisesRegex(ValueError, "equal full sequences"):
            provider.estimate(query(self.shape, (8, 16), "prefill"))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            provider.estimate(replace(query(self.shape, (16,)), context_lengths=()))

    def test_tp_cache_replicates_and_weights_shard(self):
        plan = configuration(a_tp=2)[4]
        partitions = kv_head_partitions(num_kv_heads=1, tensor_parallel_size=2, head_dim=192,
            num_layers=61, block_size=16, dtype_size_bytes=2, mla_cache_width=576)
        for partition in partitions:
            self.assertEqual(partition.bytes_per_token, 576*2*61)
            self.assertEqual(partition.bytes_per_block, 576*2*61*16)
            self.assertEqual(partition.replica_factor, 2)
        for binding in plan.rank_bindings:
            if binding.capability == "af_attention":
                self.assertEqual(binding.kv_partition.bytes_per_token, 576*2)
        expected = self.shape.mla_dimensions.weight_elements(64, 7168)*2
        attention = [c for p in plan.physical_states for c in p.components if c.category == "attention"]
        self.assertEqual(len(attention), 2)
        self.assertTrue(all(c.bytes == expected for c in attention))
        embeddings = [c for p in plan.physical_states for c in p.components if c.category == "embedding_lm_head"]
        self.assertEqual(sum(c.bytes for c in embeddings), 129280*7168*2*2)

    def test_full_model_capacity_and_dense_layer_layout(self):
        plan = configuration(full_model=True)[4]
        self.assertEqual(len(plan.layers), 61)
        self.assertEqual([p.layer_type for p in plan.layers[:4]], ["dense", "dense", "dense", "moe"])
        self.assertEqual(sum(p.layer_type == "moe" for p in plan.layers), 58)
        self.assertTrue(all(p.model_bytes < p.capacity_bytes for p in plan.physical_states))
        self.assertTrue(all(p.capacity_bytes == ASSUMPTIONS["memory_capacity_bytes"] for p in plan.physical_states))
        self.assertGreater(sum(p.model_bytes for p in plan.physical_states), 1.3e12)
        with self.assertRaisesRegex(Exception, "capacity"):
            configuration(full_model=True, f_tp=2, a_tp=2)

    def test_tp_two_runs_real_decode_and_rejects_oversize_cache(self):
        config = configuration(a_tp=2, af_batch_size=1)
        discovery = ShapeDiscovery(ASSUMPTIONS)
        simulate(discovery, config, 2, 16)
        cores = [s["key"] for s in discovery.shapes.values() if s["table"] == "mla_attention"]
        self.assertTrue(cores)
        self.assertTrue(all(k["num_heads"] == 64 and k["batch_size"] == 1 for k in cores))
        with tempfile.TemporaryDirectory() as temp:
            lookup = OperatorDataLookup(generate_package(temp, discovery, ASSUMPTIONS))
            executor, _ = simulate(lookup, config, 2, 16)
            self.assertTrue(any(e.event_type == "collective" for e in executor.trace.events))
        with self.assertRaisesRegex(ValueError, "KV cache exceeds"):
            simulate(discovery, config, 100000, 100000)

    def test_offline_package_decode_trace_missing_data_and_measured_override(self):
        discovery = ShapeDiscovery(ASSUMPTIONS)
        simulate(discovery, self.config, 2, 16)
        provider = OperatorDataComputeLatencyProvider(discovery,
            ModelOperatorManifest({self.shape.model_id: self.shape}), "analytical")
        provider.estimate(query(self.shape, (8,), "prefill"))
        core = next(s for s in discovery.shapes.values() if s["table"] == "mla_attention" and s["key"]["phase"] == "decode")
        with tempfile.TemporaryDirectory() as temp:
            package = generate_package(Path(temp)/"analytical", discovery, ASSUMPTIONS)
            lookup = OperatorDataLookup(package)
            analytical = lookup.lookup("mla_attention", core["key"])
            self.assertEqual(analytical.values["timing_method"], "analytical")
            executor, rows = simulate(lookup, self.config, 2, 16)
            self.assertGreater(rows[0]["makespan_us"], 0)
            stages = {e.stage for e in executor.trace.events}
            self.assertTrue({"attention", "routed_expert", "shared_expert", "weighted_combine", "residual"} <= stages)
            self.assertTrue(any(e.direction == "a_to_f" for e in executor.trace.events))
            self.assertTrue(any(e.direction == "f_to_a" for e in executor.trace.events))
            for mutation in (dict(kv_lora_rank=256), dict(context_lengths="17,17"), dict(v_head_dim=64)):
                with self.assertRaises(MissingOperatorDataError):
                    lookup.lookup("mla_attention", dict(core["key"], **mutation))
            source = SourceRecord("fixture:measured-core", EvidenceGrade.A, "measured", HW.device,
                HW.backend, "synthetic test observation, not a device measurement", "fixture")
            package = generate_package(Path(temp)/"measured", discovery, ASSUMPTIONS,
                [Observation("mla_attention", core["key"], 123.0, source.source_id)], [source])
            measured = OperatorDataLookup(package).lookup("mla_attention", core["key"])
            self.assertEqual(measured.latency_us, 123.0)
            self.assertEqual(measured.source_id, source.source_id)
            self.assertEqual(measured.values["timing_method"], "measured")
            self.assertIsNone(measured.values["fixed_overhead_us"])
            self.assertFalse(measured.values["calibration_applied"])


if __name__ == "__main__":
    unittest.main()
