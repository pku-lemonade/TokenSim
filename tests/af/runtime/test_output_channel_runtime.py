"""Native memory, decode and prefill for context-parallel output fragments."""

from dataclasses import replace
from pathlib import Path
import unittest

import simpy

from TokenSim.af.runtime.kv_sequence import attention_partition
from TokenSim.af.runtime.lpx_memory import workspace_components
from TokenSim.af.serving.service import AFService
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration


class OutputChannelRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[3] / "data"
        )
        cls.model = replace(
            cls.hardware.model("gemma-4-31b-text"),
            hidden_size=64,
            intermediate_size=128,
            num_layers=1,
            attention_windows=(0,),
            attention_geometries=(),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
        )

    def runtime(self, model=None):
        return configuration(
            self.hardware,
            model or self.model,
            Layout(
                a_device="groq_lp30",
                f_device="groq_lp30",
                a_tp=2,
                a_cp=8,
                f_tp=16,
                f_ep=1,
            ),
            namespace="channel",
            colocated_dense_reuse=True,
        )

    def forward(self, phase):
        runtime = self.runtime()
        env = simpy.Environment()
        service = AFService(env, runtime, seed=42, retain_trace=True)
        session = service.admit(
            0, 0, maximum_context=64, current_context=0 if phase == "prefill" else 31
        )
        kwargs = dict(query_tokens=8) if phase == "prefill" else {}
        env.run(until=env.process(service.forward([session], phase=phase, **kwargs)))
        self.assertGreater(env.now, 0)
        return runtime, session, service.trace.events

    def test_shape_weights_and_shared_kv_ownership(self):
        plan = self.runtime().plan
        partition = attention_partition(self.model, 0, 16, 8)
        self.assertEqual(
            (
                partition.query_heads_per_rank,
                partition.output_heads_per_rank,
                partition.output_head_dim,
            ),
            (4, 1, 2),
        )
        self.assertEqual(partition.as_dict()["output_head_dim"], 2)
        for binding in plan.rank_bindings:
            if binding.domain != "channel-a-domain":
                continue
            shape = plan.local_shape(
                binding.domain, binding.domain_rank, 0, "attention"
            )
            self.assertEqual(
                (shape["num_heads"], shape["head_dim"], shape["query_width"]),
                (4, 4, 16),
            )
            self.assertEqual((shape["output_heads"], shape["output_head_dim"]), (1, 2))
            self.assertEqual(shape["output_channel_start"], binding.tp_rank * 2)
            state = plan.physical_state(binding.physical_identity)
            component = next(c for c in state.components if c.category == "attention")
            # Replicated full Q/K/V; only two O input channels belong to this rank.
            weight_size = dtype_size_bytes(
                plan.effective_dtypes.weight_dtype, role="weight_dtype"
            )
            norm_elements = 2 * self.model.head_dim
            self.assertEqual(
                component.bytes, (64 * (16 + 2 + 2 * 4) + norm_elements) * weight_size
            )
            self.assertEqual(dict(component.shape)["o_width"], 2)
        kv_heads = [
            b.kv_partition.for_layer(0) for b in plan.rank_bindings if b.kv_partition
        ]
        self.assertEqual(sum(p.local_head_count for p in kv_heads), 16)
        self.assertEqual({p.kv_parallel_size for p in kv_heads}, {8})

    def test_decode_retains_full_attention_and_nonzero_output_projection(self):
        _, session, events = self.forward("decode")
        self.assertEqual(session.cached_context, 32)
        combines = [e for e in events if e.operation == "all_to_all"]
        self.assertEqual(len(combines), 2)
        self.assertEqual({e.payload_bytes for e in combines}, {8 * (2 + 1) * 4})
        outputs = [e for e in events if e.operation == "attention_output"]
        self.assertEqual(len(outputs), 16)
        for event in outputs:
            merge, project = event.details["components"]
            self.assertEqual(merge["name"], "lse_merge")
            self.assertEqual(
                (merge["key"]["num_tokens"], merge["key"]["hidden_size"]), (8, 3)
            )
            self.assertEqual(project["key"]["k"], 2)
        for event in events:
            if event.operation == "attention":
                core = next(
                    c
                    for c in event.details["components"]
                    if c["name"] == "attention_core"
                )
                self.assertEqual(
                    (core["key"]["num_heads"], core["key"]["head_dim"]), (4, 4)
                )

    def test_complete_prefill_uses_fragment_payload(self):
        _, session, events = self.forward("prefill")
        self.assertEqual(session.cached_context, 8)
        combine = [e for e in events if e.operation == "kv_combine"]
        self.assertTrue(combine)
        self.assertEqual(
            {e.details["payload_bytes"] for e in combine}, {8 * 8 * (2 + 1) * 4}
        )

    def test_fragment_scratch_reserves_duplicated_lse_in_both_directions(self):
        runtime = self.runtime()
        component = next(
            c
            for c in runtime.plan.physical_states[0].components
            if c.category == "attention"
        )
        policy = dict(
            prefill_chunk=1, maximum_context=64, attention_tile_keys=16, tile_buffers=2
        )
        buffers = dict(workspace_components((component,), self.model, policy))
        # QKV + scores + online state + cache tiles + send/receive fragment packets.
        expected = (
            (16 + 2 * 4) * 2
            + 4 * 8 * 4
            + (16 + 2 * 4) * 4
            + 2 * 8 * 2 * 4 * (2 + 2)
            + 2 * 8 * (2 + 1) * 4
        )
        self.assertEqual(buffers["lp30_live_scratch"], expected)

    def test_unselected_local_layers_and_uneven_fragments_remain_rejected(self):
        for model in (
            replace(self.model, attention_windows=(8,)),
            replace(self.model, head_dim=3),
        ):
            with self.assertRaises(ConfigurationError):
                self.runtime(model)


if __name__ == "__main__":
    unittest.main()
