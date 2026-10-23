"""KV-parallel memory, decode combine graph, costs and complete prefill."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.providers.context_parallel import combine_payload_bytes
from TokenSim.af.runtime.admission import rank_layer_kv_bytes
from TokenSim.af.runtime.kv_sequence import (
    validate_kv_parallel_domain,
    visible_tokens,
)
from TokenSim.af.serving.service import AFService
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration, runtime_configurations


class KVParallelRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[3] / "data"
        )
        # Layer 0 uses a window and stays on TP under full_attention selection.
        cls.model = replace(
            cls.hardware.model("gemma-4-31b-text"),
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            attention_windows=(8, 0),
            attention_geometries=(),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
        )

    def runtime(self, kvp, *, layers="full_attention", block=4, edit=None):
        def build(doc, **kwargs):
            if edit:
                edit(doc)
            return build_model_runtime(doc, **kwargs)

        layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=4 // kvp,
            a_cp=kvp,
            f_tp=4,
            f_ep=1,
        )
        with patch("scripts.lpx.native_config.build_model_runtime", build):
            return configuration(
                self.hardware,
                self.model,
                layout,
                namespace="kvp",
                colocated_dense_reuse=True,
                kv_parallel_layers=layers,
                kv_append_block=block,
            )

    def decode(self, runtime, context):
        env = simpy.Environment()
        service = AFService(env, runtime, seed=42, retain_trace=True)
        session = service.admit(0, 0, maximum_context=64, current_context=context)
        env.run(until=env.process(service.forward([session], phase="decode")))
        return env.now, service.trace.events

    def test_owned_kv_sums_to_unreplicated_bytes_on_selected_layers(self):
        context = 37
        for kvp in (1, 2, 4):
            plan = self.runtime(kvp).plan
            totals = [0, 0]
            for binding in plan.rank_bindings:
                if binding.kv_partition is None:
                    continue
                for layer, size in rank_layer_kv_bytes(plan, binding, context).items():
                    totals[layer] += size
            per_head = self.model.head_dim * 2 * plan.kv_dtype_size_bytes
            # Two KV heads replicate over attention TP ranks beyond the head count.
            replicas = max(1, 4 // kvp // 2)
            with self.subTest(kvp=kvp):
                self.assertEqual(totals[1], 2 * per_head * context * replicas)
                self.assertEqual(totals[0], 2 * per_head * 8 * 2)

    def test_replicated_projection_weights_and_plan_summary(self):
        baseline, split = self.runtime(1).plan, self.runtime(2).plan

        def component(plan, layer):
            state = plan.physical_states[0]
            return next(
                c
                for c in state.components
                if c.component_id.startswith(f"layer.{layer}.attention")
            )

        hidden, d = 32, 4
        old, new = component(baseline, 1), component(split, 1)
        self.assertEqual(dict(new.shape)["q_heads_local"], 4)
        self.assertEqual(dict(new.shape)["o_heads_local"], 2)
        # Q/K/V keep one KV head per rank and add two query heads.
        weight = dtype_size_bytes(self.model.dtype, role="weight_dtype")
        self.assertEqual(new.bytes - old.bytes, 2 * d * hidden * weight)
        self.assertEqual(component(baseline, 0), component(split, 0))
        domains = {d["name"]: d for d in split.summary()["domains"]}
        self.assertEqual(domains["kvp-a-domain"]["kv_parallel_size"], 2)
        self.assertNotIn(
            "kv_parallel_size",
            {k for d in baseline.summary()["domains"] for k in d},
        )
        partition = split.rank_bindings[1].kv_partition.as_dict()
        self.assertEqual(partition["layer_partitions"]["1"]["kv_parallel_rank"], 1)
        self.assertNotIn("kv_parallel_size", partition["layer_partitions"]["0"])

    def test_combine_graph_payload_and_local_contexts(self):
        for context in (20, 40):
            _, events = self.decode(self.runtime(2), context)
            combine = [e for e in events if e.operation == "all_to_all"]
            outputs = [e for e in events if e.operation == "attention_output"]
            self.assertEqual(len(combine), 2)
            self.assertEqual(len(outputs), 4)
            self.assertEqual(
                {e.payload_bytes for e in combine}, {combine_payload_bytes(1, 2, 4, 2)}
            )
            for event in events:
                if event.operation != "attention" or event.layer_id != 1:
                    continue
                core = next(
                    c
                    for c in event.details["components"]
                    if c["name"] == "attention_core"
                )
                rank = event.details["local_shape"]["kv_parallel_rank"]
                self.assertEqual(
                    core["key"]["context_len"],
                    visible_tokens(context + 1, None, 2, rank, 4),
                )
                names = [c["name"] for c in event.details["components"]]
                self.assertNotIn("output_projection", names)

    def test_output_node_merges_fragments_before_projection(self):
        _, events = self.decode(self.runtime(2), 60)
        for event in events:
            if event.operation == "attention_output":
                names = [c["name"] for c in event.details["components"]]
                self.assertEqual(names, ["lse_merge", "output_projection"])
                merge = event.details["components"][0]["key"]
                self.assertEqual(
                    (merge["num_tokens"], merge["hidden_size"]), (1 * 2 * 2, 4 + 1)
                )

    def test_all_layer_selection_shards_windowed_layers(self):
        plan = self.runtime(2, layers="all").plan
        partition = plan.rank_bindings[0].kv_partition.for_layer(0)
        self.assertEqual(partition.kv_parallel_size, 2)
        _, events = self.decode(self.runtime(2, layers="all"), 30)
        self.assertEqual(len([e for e in events if e.operation == "all_to_all"]), 2 * 2)

    def test_complete_prefill_records_combine(self):
        env = simpy.Environment()
        service = AFService(env, self.runtime(2), seed=42, retain_trace=True)
        session = service.admit(0, 0, maximum_context=64, current_context=0)
        env.run(
            until=env.process(
                service.forward([session], phase="prefill", query_tokens=16)
            )
        )
        kinds = {e.operation for e in service.trace.events}
        self.assertIn("kv_combine", kinds)
        self.assertEqual(session.cached_context, 16)

    def test_unsupported_scopes_raise_configuration_errors(self):
        domain = self.runtime(2).cluster.af_domains[0]
        for colocated, mla in ((False, False), (True, True)):
            with self.assertRaises(ConfigurationError):
                validate_kv_parallel_domain(
                    domain, self.model, colocated=colocated, mla=mla
                )
        with self.assertRaises(ConfigurationError):
            Layout(a_tp=2, a_cp=2, f_tp=2, f_ep=1, a_device="rubin")
        split = Layout(
            a_device="groq_lp30", f_device="groq_lp30", a_tp=2, a_cp=2, f_tp=4, f_ep=1
        )
        self.assertEqual((split.a_stage_ranks, split.a_ranks), (4, 4))
        for case in (
            SimpleNamespace(decode=split, prefill=split, draft=None),
            SimpleNamespace(decode=split, prefill=None, draft=split),
        ):
            with self.assertRaisesRegex(ConfigurationError, "handoff"):
                runtime_configurations(self.hardware, self.model, case)
        with self.assertRaises(ConfigurationError):
            configuration(
                self.hardware,
                self.model,
                Layout(a_device="rubin", f_device="rubin", a_tp=2, a_cp=2, f_tp=4),
                namespace="kvp",
            )


if __name__ == "__main__":
    unittest.main()
