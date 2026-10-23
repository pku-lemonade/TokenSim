"""Independent causal and resource oracles for LPX output-vector overlap."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import simpy

from TokenSim.comm.collective_types import CollectiveEstimate, LevelEstimate
from TokenSim.comm.streaming import StreamingConfig, output_vectors
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.latency_ledger import build, catalog_collective
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import NativeProfile, cached_profile, capture_profile
from scripts.lpx.streaming import StreamingBuilder, duration, transform
from TokenSim.af.providers.traffic import TransferPlanMixin


class VectorGeometryTest(unittest.TestCase):
    def estimate(self):
        return CollectiveEstimate(
            9,
            "all_reduce",
            2,
            10,
            "analytical",
            "test",
            (LevelEstimate("tray", 2, "direct", 2, 5, "link", 6),),
            3,
        )

    def test_tail_alignment_wire_conservation_and_single_launch(self):
        vectors = output_vectors(10, 2, 4, self.estimate(), 1)
        self.assertEqual([v.offset_bytes for v in vectors], [0, 4, 8])
        self.assertEqual([v.size_bytes for v in vectors], [4, 4, 2])
        self.assertAlmostEqual(sum(v.production_fraction for v in vectors), 1)
        self.assertAlmostEqual(sum(v.wire_bytes_per_phase for v in vectors), 5)
        self.assertAlmostEqual(sum(v.wire_seconds_per_phase for v in vectors), 2e-6)

    def test_unsupported_or_inconsistent_estimates_fail(self):
        estimate = self.estimate()
        cases = [
            replace(estimate, match_type="measured"),
            replace(estimate, levels=estimate.levels * 2),
            replace(estimate, levels=(replace(estimate.levels[0], algorithm="ring"),)),
            replace(estimate, latency_us=8),
            replace(estimate, levels=(replace(estimate.levels[0], bytes_per_step=4),)),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ConfigurationError):
                output_vectors(10, 2, 4, case, 1)
        for payload, element, vector in ((0, 2, 4), (9, 2, 4), (10, 2, 3)):
            with self.assertRaises(ConfigurationError):
                output_vectors(payload, element, vector, estimate, 1)

    def test_config_rejects_invalid_values(self):
        for value in (0, -1, True, 0.5):
            with self.assertRaises(ConfigurationError):
                StreamingConfig(vector_bytes=value)
        for value in (-1, float("nan"), True):
            with self.assertRaises(ConfigurationError):
                StreamingConfig(collective_launch_us=value)
        with self.assertRaises(ConfigurationError):
            StreamingConfig(overlap=1)
        with self.assertRaises(ConfigurationError):
            StreamingConfig(early_output=1)


class CausalStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )

    def fixture(self, blocked=False):
        runtime = SimpleNamespace(
            hardware=self.hardware,
            workers=[
                SimpleNamespace(
                    hardware="groq_lp30",
                    physical_worker_ref=SimpleNamespace(
                        identity=f"g:{i}", worker_index=i
                    ),
                )
                for i in range(2)
            ],
        )
        events = [
            SimpleNamespace(
                node_id="p0", operation="attention", physical_endpoints=("g:0",)
            ),
            SimpleNamespace(
                node_id="p1", operation="attention", physical_endpoints=("g:1",)
            ),
            SimpleNamespace(
                node_id="ar",
                operation="all_reduce",
                physical_endpoints=("g:0", "g:1"),
                predecessor_ids=("p0", "p1"),
                payload_bytes=640,
                match_type="analytical",
                details={"wire_dtype": "bf16"},
            ),
            SimpleNamespace(node_id="consumer", operation="rmsnorm"),
        ]
        nodes = [
            ReplayNode((), ("w0",), "w0", 0, 12e-6, 12e-6, 0),
            ReplayNode((), ("w1",), "w1", 0, 15e-6, 15e-6, 1),
            ReplayNode((0, 1), ("wire",), "wire", 0, 9e-6, 9e-6, 2),
            ReplayNode((2,), ("w0",), "w0", 0, 1e-6, 1e-6, 3),
        ]
        if blocked:
            nodes.append(ReplayNode((), ("wire",), "wire", 0, 20e-6, 20e-6, 0))
            events.append(SimpleNamespace(node_id="blocker", operation="transfer"))
        profile = NativeProfile(tuple(nodes), dict(phase="decode", batch=1, queries=1))
        return runtime, events, profile

    def oracle(self, blocked=False, early=True):
        # SimPy producers hold a worker continuously, independent of replay DAG splitting.
        env = simpy.Environment()
        wire = simpy.Resource(env, capacity=1)
        ready = [[env.event(), env.event()] for _ in range(2)]

        def producer(rank, prefix):
            yield env.timeout(prefix)
            for k in range(2):
                yield env.timeout(5e-6)
                ready[rank][k].succeed()

        for rank, prefix in enumerate((2e-6, 5e-6)):
            env.process(producer(rank, prefix))

        def occupy():
            with wire.request() as request:
                yield request
                yield env.timeout(20e-6)

        if blocked:
            env.process(occupy())

        def launch():
            yield env.all_of([rank[0 if early else 1] for rank in ready])
            yield env.timeout(0.5e-6)

        launched = env.process(launch())

        def vector(k):
            yield env.all_of([launched, *(rank[k] for rank in ready)])
            for _ in range(2):
                with wire.request() as request:
                    yield request
                    # 640 bytes * (2-1)/2 / 2 vectors / 1.25 TB/s.
                    yield env.timeout(160 / 1.25e12)
                yield env.timeout(0.05e-6)

        def consumer():
            yield env.all_of([env.process(vector(k)) for k in range(2)])
            yield env.timeout(1e-6)

        env.process(consumer())
        env.run()
        return env.now

    def test_slow_rank_and_shared_injection_match_independent_oracle(self):
        for blocked in (False, True):
            for early in (False, True):
                with self.subTest(blocked=blocked, early=early):
                    runtime, events, profile = self.fixture(blocked)
                    builder = StreamingBuilder(
                        runtime, events, profile, StreamingConfig(early_output=early)
                    )
                    with patch.object(
                        builder, "producer_work", return_value=(10e-6, 2e-6)
                    ):
                        nodes, metadata = builder.build()
                    self.assertAlmostEqual(
                        duration(nodes), self.oracle(blocked, early), places=14
                    )
                    kernel = ReplayKernel()
                    try:
                        kernel.window(0)
                        kernel.submit(kernel.add(nodes), 0)
                        kernel.advance()
                        stats = kernel.stats()
                    finally:
                        kernel.close()
                    self.assertAlmostEqual(stats["w0"][0], 13e-6, places=14)
                    self.assertAlmostEqual(stats["w1"][0], 15e-6, places=14)
                    self.assertAlmostEqual(
                        stats["wire"][0],
                        640 / 1.25e12 + (20e-6 if blocked else 0),
                        places=14,
                    )
                    self.assertEqual(metadata["injected_bytes_per_endpoint_sum"], 640)
                    self.assertGreaterEqual(duration(nodes), 15e-6 + 1e-6 + 0.1e-6)

    def test_whole_buffer_equals_catalog_and_earlier_output_reduces_latency(self):
        runtime, events, profile = self.fixture()
        results = []
        for config in (
            StreamingConfig(overlap=False),
            StreamingConfig(vector_bytes=640),
            StreamingConfig(),
        ):
            builder = StreamingBuilder(runtime, events, profile, config)
            with patch.object(builder, "producer_work", return_value=(10e-6, 2e-6)):
                nodes, _ = builder.build()
            results.append(duration(nodes))
        self.assertEqual(results[0], results[1])
        self.assertLess(results[2], results[0])


class NativeStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware = HardwareContext.load(Path(__file__).resolve().parents[1] / "data")
        cls.runtime = build(hardware)
        cls.settings = dict(
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=10750,
            dp_rank=0,
            seed=42,
            sample=0,
        )

    def test_native_catalog_control_and_compute_resource_conservation(self):
        native = capture_profile(self.runtime, "ledger", **self.settings)
        catalog = capture_profile(
            self.runtime, "ledger", streaming={"overlap": False}, **self.settings
        )
        with patch.object(
            TransferPlanMixin, "_collective", catalog_collective(self.runtime)
        ):
            aligned = capture_profile(self.runtime, "ledger", **self.settings)
        self.assertAlmostEqual(
            duration(catalog.nodes), duration(aligned.nodes), places=13
        )
        overlap = capture_profile(self.runtime, "ledger", streaming={}, **self.settings)
        self.assertLess(duration(overlap.nodes), duration(catalog.nodes))
        busy = []
        for profile in (native, catalog, overlap):
            totals = {}
            for node in profile.nodes:
                for resource in node.resources:
                    if resource.startswith("worker:"):
                        totals[resource] = totals.get(resource, 0) + node.occupancy
            busy.append(totals)
        self.assertEqual(set(busy[0]), set(busy[2]))
        for resource in busy[0]:
            self.assertAlmostEqual(busy[0][resource], busy[1][resource], places=13)
            self.assertAlmostEqual(busy[0][resource], busy[2][resource], places=13)
        self.assertEqual(
            overlap.metadata["streaming"]["streaming_collectives"],
            2 * self.runtime.model.num_layers,
        )

    def test_streaming_cache_roundtrip_and_mode_identity(self):
        with tempfile.TemporaryDirectory() as cache:
            profiles = [
                cached_profile(
                    self.runtime, "ledger", cache, streaming=settings, **self.settings
                )
                for settings in ({"overlap": False}, {}, {})
            ]
        self.assertEqual(profiles[1].nodes, profiles[2].nodes)
        self.assertNotEqual(
            profiles[0].metadata["input_sha256"], profiles[1].metadata["input_sha256"]
        )
        self.assertIn("modeled_duration_s", profiles[1].metadata)
        profiles[2].verify()

    def test_prefill_is_explicitly_skipped_and_multitoken_decode_fails(self):
        profile = NativeProfile((), dict(phase="prefill", batch=1, queries=32))
        skipped = transform(self.runtime, (), profile, {})
        self.assertEqual(skipped.metadata["streaming_skipped"], "prefill")
        with self.assertRaises(ConfigurationError):
            transform(
                self.runtime,
                (),
                replace(profile, metadata=dict(phase="decode", batch=1, queries=2)),
                {},
            )

    def test_batched_decode_streams_every_request_row(self):
        hardware = HardwareContext.load(Path(__file__).resolve().parents[1] / "data")
        runtime = build(hardware, micro_batch=2)
        settings = {**self.settings, "batch": 2, "context": 2000}
        single = capture_profile(runtime, "ledger", streaming={}, **settings)
        mixed = capture_profile(
            runtime,
            "ledger",
            streaming={},
            **{**settings, "contexts": (2000, 6000), "maximum_contexts": (2000, 6000)},
        )
        for profile in (single, mixed):
            self.assertEqual(
                profile.metadata["streaming"]["streaming_collectives"],
                2 * runtime.model.num_layers,
            )
        self.assertNotIn("contexts", single.metadata)
        self.assertEqual(mixed.metadata["contexts"], [2000, 6000])
        self.assertGreater(
            mixed.metadata["modeled_duration_s"], single.metadata["modeled_duration_s"]
        )
        with self.assertRaises(ConfigurationError):
            capture_profile(runtime, "ledger", **{**settings, "contexts": (2000,)})


if __name__ == "__main__":
    unittest.main()
