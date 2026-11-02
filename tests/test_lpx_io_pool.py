"""I/O-only placement has independent storage and exact graph dependencies."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import ConfigurationError
from scripts.lpx.fidelity import load
from scripts.run_lpx_article_sweep import article_config
from scripts.run_lpx_context_parallel import environment
from scripts.lpx.io_pool import (
    column_shards,
    pool_memory,
    transformer_component,
    transformer_setup,
)
from scripts.lpx.program import LP30Program
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import NativeProfile, save_profile
from scripts.bound_lpx_mfu_layouts import saved_profile


class IOPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = article_config(
            load(
                Path(__file__).resolve().parents[1]
                / "experiments/lpx/fidelity-study.json"
            )
        )
        cls.config = config
        cls.hardware, cls.model, _, _ = environment(config)

    def test_proxy_keeps_communication_budget_and_original_config(self):
        config, hardware, model, proxy = transformer_setup(
            self.config, tp=16, queries=3
        )
        self.assertEqual(proxy.vocab_size, 16)
        self.assertEqual(model, self.model)
        self.assertEqual(
            config["memory"]["reserve_bytes"],
            self.config["memory"]["reserve_bytes"]
            + 12 * (model.vocab_size - model.hidden_size),
        )
        self.assertEqual(config["memory"]["prefill_chunk"], 3)
        self.assertEqual(self.config["memory"]["prefill_chunk"], 16)
        self.assertEqual(
            hardware.device("groq_lp30").memory_capacity_bytes.value, 500_000_000
        )

    def test_three_way_columns_preserve_every_vocab_entry_and_pool_capacity(self):
        shards = column_shards(self.model.vocab_size, 3)
        self.assertEqual(sum(shards), self.model.vocab_size)
        self.assertEqual(max(shards) - min(shards), 1)
        rows = pool_memory(self.model, 3, 1)
        self.assertTrue(all(row["headroom_bytes"] > 0 for row in rows))
        self.assertEqual(
            sum(row["weights_bytes"] for row in rows),
            self.model.hidden_size * self.model.vocab_size
            + 3 * 2 * self.model.hidden_size,
        )
        with self.assertRaises(ConfigurationError):
            pool_memory(self.model, 2, 1)

    def test_include_reindexes_forward_edges_and_lifetime_release(self):
        p = LP30Program(self.hardware, 1)
        parent = p.append((), duration=3)
        source = NativeProfile(
            (
                ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=1),
                ReplayNode((2,), (), None, 0, 1, 0),
                ReplayNode((0,), (), None, 0, 2, 0),
            ),
            {},
        )
        end = p.include(source, (parent,))
        self.assertEqual(p.nodes[1].release_after, 2)
        self.assertEqual(p.nodes[2].predecessors, (3,))
        kernel = ReplayKernel()
        self.addCleanup(kernel.close)
        identifier = kernel.add(p.nodes)
        kernel.submit(identifier, 0)
        self.assertEqual(kernel.advance().end, 6)
        self.assertEqual(end, 4)

    def test_single_participant_reduction_has_only_its_input_dependency(self):
        p = LP30Program(self.hardware, 1)
        parent = p.append((), duration=2)
        result = p.reduce((parent,), (0,), 64)
        self.assertEqual(p.nodes[result].predecessors, (parent,))
        self.assertEqual(p.nodes[result].resources, ())
        self.assertEqual(p.nodes[result].duration, 0)
        with self.assertRaises(ConfigurationError):
            p.reduce((parent,), (), 64)

    def test_boundary_elision_preserves_transformer_work_and_its_dependencies(self):
        profile = NativeProfile(
            (
                ReplayNode((), ("w0",), "w0", 0, 1, 1),
                ReplayNode((), ("w1",), "w1", 0, 1, 1),
                ReplayNode((5,), (), None, 0, 0, 0),
                ReplayNode((2,), ("w0",), "w0", 0, 3, 3),
                ReplayNode((0, 1), ("wire",), "wire", 0, 1, 1),
                ReplayNode((4,), (), None, 2, 0, 0),
            ),
            dict(
                native_node_ids=[
                    "i:b:boundary:embedding_lookup:r0",
                    "i:b:boundary:embedding_lookup:r1",
                    "i:b:boundary:embedding_reduce",
                    "i:b:l0:attention",
                ],
                input_sha256="original",
                node_body_sha256="original-body",
                modeled_duration_s=7,
            ),
        )
        with tempfile.TemporaryDirectory() as cache:
            component = transformer_component(profile, tp=2, cache=cache)
            self.assertEqual(component.metadata["modeled_duration_s"], 3)
            self.assertEqual(component.nodes[3], profile.nodes[3])
            self.assertEqual(
                component.metadata["elided_boundary_nodes"], [0, 1, 2, 4, 5]
            )
            with self.assertRaises(ConfigurationError):
                transformer_component(profile, tp=4, cache=cache)

    def test_lifetime_snapshot_reconstructs_without_a_native_capture_duration(self):
        nodes = (
            ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=2),
            ReplayNode((0,), (), None, 0, 2, 0),
            ReplayNode((1,), (), None, 1, 0, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json.gz"
            saved = save_profile(
                path, NativeProfile(nodes, dict(modeled_duration_s=3)), dict(kind="toy")
            )
            restored = saved_profile(path, saved.metadata["input_sha256"])
            self.assertEqual(restored.nodes, nodes)
            restored.verify()


if __name__ == "__main__":
    unittest.main()
