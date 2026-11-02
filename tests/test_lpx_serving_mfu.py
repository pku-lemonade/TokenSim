"""Exact request work, bounded-stage wavefronts and terminal-only logits."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from scripts.lpx.fidelity import load
from scripts.lpx.context_envelopes import reserve_contexts
from scripts.lpx.io_pool import pool_memory, transformer_setup
from scripts.lpx.mfu_metrics import useful_matrix_flops
from scripts.lpx.profile_stages import split_stages
from scripts.lpx.replay_engine import ReplayNode
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.serving_metrics import forward_work, key_pairs, serving_work
from scripts.lpx.serving_pipeline import ServingPipeline, ServingWorkload
from scripts.run_lpx_article_sweep import article_config


class ToyBank:
    pp = 2

    def key(self, kind, task, stage):
        return kind, stage

    def profile(self, kind, task, stage):
        duration = {
            "input": 1,
            "entry": 0,
            "body": 2,
            "handoff": 1,
            "head_copy": 1,
            "head": 1,
        }[kind]
        resource = f"stage:{stage}" if kind in {"entry", "body"} else kind
        return NativeProfile(
            (ReplayNode((), (resource,), resource, 0, duration, duration),), {}
        )

    def admit(self, *args):
        pass

    def release(self, *args):
        pass

    def commit(self, *args):
        pass

    def validate_empty(self):
        pass


class ServingMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = article_config(
            load(
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/fidelity-study.json"
            )
        )
        _, _, cls.model, _ = transformer_setup(cls.config, tp=16, queries=1)

    def test_exact_causal_pairs_match_integer_sum(self):
        for window in (0, 1, 4, 16):
            for lo, hi in ((1, 1), (1, 7), (3, 19)):
                self.assertEqual(
                    key_pairs(lo, hi, window),
                    sum(min(i, window or i) for i in range(lo, hi + 1)),
                )

    def test_request_work_is_independent_of_chunking_and_logits_are_terminal(self):
        model = self.model
        inputs = 17
        outputs = 4
        expected = serving_work(model, inputs=inputs, outputs=outputs)
        total = 0
        for start in range(0, inputs, 5):
            n = min(5, inputs - start)
            total += forward_work(
                model, offset=start, queries=n, logits_rows=int(start + n == inputs)
            )["total"]
        for context in range(inputs + 1, inputs + outputs):
            total += useful_matrix_flops(model, context)["total"]
        self.assertEqual(total, expected["total"])
        self.assertEqual(expected["logits_rows"], outputs)
        self.assertEqual(expected["decode_forwards"], outputs - 1)
        head = 2 * model.hidden_size * model.vocab_size
        self.assertEqual(expected["prefill"]["logits"], head)
        self.assertEqual(
            serving_work(model, inputs=inputs, outputs=1)["decode"]["total"], 0
        )

    def test_prefill_boundary_memory_uses_only_one_output_row(self):
        _, _, model, _ = transformer_setup(self.config, tp=16, queries=1)
        with self.assertRaises(ConfigurationError):
            pool_memory(model, 3, 256)
        rows = pool_memory(model, 3, 256, logits_rows=1)
        self.assertTrue(all(row["headroom_bytes"] > 0 for row in rows))
        updated, _, _, _ = transformer_setup(
            self.config, tp=16, queries=256, logits_rows=1
        )
        self.assertEqual(
            updated["memory"]["reserve_bytes"], self.config["memory"]["reserve_bytes"]
        )
        with self.assertRaises(ConfigurationError):
            pool_memory(model, 3, 2, logits_rows=3)


class ServingPipelineTest(unittest.TestCase):
    def test_two_stage_prompt_wavefront_has_analytic_completion_clock(self):
        result = ServingPipeline(
            ToyBank(), ServingWorkload(inputs=4, outputs=2, chunk=2)
        ).run()
        self.assertEqual(result["request_timings"][0]["first_output"], 11)
        self.assertEqual(result["seconds"], 19)
        self.assertEqual(result["decode_forwards"], 1)
        self.assertEqual(result["counters"]["head"], 2)
        self.assertEqual(result["counters"]["input"], 3)
        self.assertEqual(result["maximum_reserved_input_slots"], [1, 1])
        self.assertEqual(result["ttft_mean_s"], 11)
        self.assertEqual(result["tpot_mean_s"], 8)
        self.assertEqual(result["measured_inter_token_intervals"], 1)

    def test_multiple_clients_cycles_and_partial_tail_preserve_every_token(self):
        w = ServingWorkload(inputs=5, outputs=3, chunk=2, clients=2, cycles=2)
        result = ServingPipeline(ToyBank(), w).run()
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["input_tokens"], 20)
        self.assertEqual(result["output_tokens"], 12)
        self.assertEqual(result["counters"]["input"], 20)
        self.assertEqual(
            len({(r["client"], r["cycle"]) for r in result["request_timings"]}), 4
        )
        self.assertTrue(
            all(r["first_output"] >= r["start"] for r in result["request_timings"])
        )

    def test_single_output_has_no_decode_forward(self):
        r = ServingPipeline(
            ToyBank(), ServingWorkload(inputs=3, outputs=1, chunk=8)
        ).run()
        self.assertEqual(r["decode_forwards"], 0)
        self.assertEqual(r["counters"]["head"], 1)
        self.assertIsNone(r["tpot_mean_s"])
        self.assertEqual(r["measured_inter_token_intervals"], 0)


class StagePartitionTest(unittest.TestCase):
    def test_stage_split_rejects_backward_dependencies_and_arena_lifetimes(self):
        later = ReplayNode((), ("x/worker:1",), "x/worker:1", 0, 1, 1)
        earlier = ReplayNode((0,), ("x/worker:0",), "x/worker:0", 0, 1, 1)
        with self.assertRaisesRegex(ConfigurationError, "backward"):
            split_stages(NativeProfile((later, earlier), {}), tp=1, pp=2)
        arena = ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=1)
        with self.assertRaisesRegex(ConfigurationError, "lifetimes"):
            split_stages(NativeProfile((arena, earlier), {}), tp=1, pp=1)

    def test_forward_resource_work_is_conserved_and_handoff_flight_is_retained(self):
        nodes = (
            ReplayNode((), ("x/host:0",), "x/host:0", 0, 4, 4),
            ReplayNode((0,), ("mfu-search/worker:0",), "mfu-search/worker:0", 0, 2, 2),
            ReplayNode((1,), ("lp30:0:tx", "lp30:1:rx"), "lp30:0:tx", 0, 3, 3),
            ReplayNode((2,), (), None, 5, 0, 0),
            ReplayNode((3,), ("mfu-search/worker:1",), "mfu-search/worker:1", 0, 7, 7),
        )
        groups = split_stages(NativeProfile(nodes, {}), tp=1, pp=2)
        self.assertEqual(groups[(0, 0)].metadata["modeled_duration_s"], 2)
        self.assertEqual(groups[(0, 1)].metadata["modeled_duration_s"], 8)
        self.assertEqual(groups[(1, 0)].metadata["modeled_duration_s"], 7)
        seen = Counter()
        for part in groups.values():
            for node in part.nodes:
                for resource in node.resources:
                    seen[resource] += node.occupancy
        self.assertEqual(
            seen,
            Counter(
                {
                    "mfu-search/worker:0": 2,
                    "lp30:0:tx": 3,
                    "lp30:1:rx": 3,
                    "mfu-search/worker:1": 7,
                }
            ),
        )


class ContextEnvelopeTest(unittest.TestCase):
    def test_every_actual_context_is_priced_even_when_ceiling_is_faster(self):
        runtime = SimpleNamespace(
            hardware=SimpleNamespace(device=lambda _: object()),
            cluster=SimpleNamespace(af=SimpleNamespace(graph_device_kernel_us=0)),
            model=SimpleNamespace(attention_window=lambda _: None),
        )
        node = ReplayNode((), ("x/worker:0",), "x/worker:0", 0, 5e-6, 5e-6)
        profile = NativeProfile(
            (node,),
            dict(
                attention_keys=[
                    dict(
                        index=0,
                        layer_id=0,
                        local_shape={},
                        component=dict(
                            match_type="analytical",
                            table="context_attention",
                            key=dict(context_len=3),
                            latency_us=5,
                        ),
                    )
                ]
            ),
        )
        module = "scripts.lpx.context_envelopes"
        with (
            patch(f"{module}.analytical_execution_device", return_value=object()),
            patch(
                f"{module}.estimate_lp30",
                side_effect=lambda _, key, device: SimpleNamespace(
                    latency_us={1: 8, 2: 12, 3: 5}[key["context_len"]]
                ),
            ),
        ):
            result = reserve_contexts(profile, runtime, (1, 2, 3), 1)
        self.assertAlmostEqual(result.nodes[0].duration, 12e-6)
        self.assertAlmostEqual(result.nodes[0].occupancy, 12e-6)
        self.assertEqual(profile.nodes[0].duration, 5e-6)
        receipt = result.metadata["context_reservations"][0]
        self.assertEqual(receipt["extension_context"], 2)
        self.assertEqual(receipt["covered_contexts"], 3)
        with self.assertRaises(ConfigurationError):
            reserve_contexts(profile, runtime, (0,), 1)


if __name__ == "__main__":
    unittest.main()
