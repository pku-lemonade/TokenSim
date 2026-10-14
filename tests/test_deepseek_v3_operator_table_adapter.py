import unittest
from dataclasses import replace

from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.device import DeviceSpec
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
from TokenSim.operator_data.workload import work_for_table
from scripts.deepseek_v3_runtime import configuration, simulate


SOURCE = SourceRecord(
    "measured:mla-fixture", "A", "measured", device="Fixture", backend="test"
)


def _mla_rows():
    base = dict(
        attn_dtype="bf16",
        kv_cache_dtype="bf16",
        phase="decode",
        execution_mode="absorbed",
        query_len=1,
        context_lengths="[[8,1]]",
        num_heads=8,
        kv_lora_rank=4,
        rope_head_dim=2,
        nope_head_dim=2,
        v_head_dim=2,
    )
    return [
        {**base, "batch_size": 1, "latency_us": 10.0, "source_id": SOURCE.source_id},
        {**base, "batch_size": 2, "latency_us": 18.0, "source_id": SOURCE.source_id},
    ]


class OperatorTableAdapterTest(unittest.TestCase):
    def test_mla_key_normalization_and_provenance(self):
        package = OperatorDataPackage.from_rows(
            PackageMeta(
                "fixture-v1", "Fixture", "test", sources={SOURCE.source_id: SOURCE}
            ),
            {"mla_attention": _mla_rows()},
        )
        lookup = OperatorLookup(package, LookupPolicy(extrapolate="none"))
        exact = lookup.lookup(
            "mla_attention", {**_mla_rows()[0], "context_lengths": "[[8, 1.0]]"}
        )
        self.assertEqual(exact.match_type, "exact")
        self.assertEqual(exact.source_id, SOURCE.source_id)
        mid = lookup.lookup("mla_attention", {**_mla_rows()[0], "batch_size": 1.5})
        self.assertEqual(mid.match_type, "interpolated")
        with self.assertRaises(MissingOperatorDataError):
            lookup.lookup("mla_attention", {**_mla_rows()[0], "batch_size": 4})

    def test_mla_work_separates_cache_read_and_write(self):
        key = _mla_rows()[0]
        work = work_for_table("mla_attention", key)
        self.assertGreater(work.details["cache_read_bytes"], 0)
        self.assertGreater(work.details["cache_write_bytes"], 0)
        self.assertEqual(
            work.kv_bytes,
            work.details["cache_read_bytes"] + work.details["cache_write_bytes"],
        )


class DeepSeekV3ModelSpecTest(unittest.TestCase):
    def test_catalog_and_mla_geometry(self):
        model = HardwareContext.load("data").model("deepseek-v3-671b-mla")
        self.assertEqual(model.num_layers, 61)
        self.assertEqual(model.moe_layer_indices()[:3], [3, 4, 5])
        self.assertEqual(model.moe_layer_indices()[-1], 60)
        self.assertEqual(model.mla_dimensions.cache_width, 576)
        self.assertEqual(model.total_params(), 671025522688)

    def test_invalid_mla_dimension_fails_at_model_construction(self):
        model = HardwareContext.load("data").model("deepseek-v3-671b-mla")
        with self.assertRaises(ConfigurationError):
            replace(model, rope_head_dim=63)


class DeepSeekV3RuntimeTest(unittest.TestCase):
    def test_full_plan_has_dense_prefix_and_rank_reuse_f_pool(self):
        config = configuration(full_model=True, af_batch_size=1)
        self.assertEqual(
            [layer.layer_type for layer in config.plan.layers[:4]],
            ["dense", "dense", "dense", "moe"],
        )
        self.assertEqual(config.plan.total_layers, 61)
        self.assertEqual(len(config.plan.rank_bindings), 33)

    def test_mla_smoke_returns_traceable_operator_profile(self):
        config = configuration(full_model=False, af_batch_size=1)
        executor, rows, compute, _ = simulate(config, 1, 8)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["makespan_us"], 0)
        self.assertTrue(
            any(event.stage == "attention" for event in executor.trace.events)
        )
        self.assertTrue(
            any(
                event.operation == "post_attention_norm"
                for event in executor.trace.events
            )
        )
        self.assertTrue(compute.stats_dict()["operator_query_count"] > 0)

    def test_tp_and_multistep_trace_has_no_orphan_frontier(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        executor, rows, _, _ = simulate(config, 2, 8, steps=2)
        self.assertEqual(len(rows), 2)
        for iteration in range(2):
            executor.trace.validate_invariants(iteration)
        self.assertTrue(
            any(event.operation == "all_reduce" for event in executor.trace.events)
        )


if __name__ == "__main__":
    unittest.main()
