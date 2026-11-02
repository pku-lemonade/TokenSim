"""Analytic oracles for exclusive buffer lifetime demand."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from TokenSim.errors import ConfigurationError
from scripts.lpx.mfu_metrics import resource_demands, throughput_bounds
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.mfu_replay import replay_cohorts


class LifetimeBoundsTest(unittest.TestCase):
    def test_explicit_program_assignment_and_offsets_preserve_feedback(self):
        fast = SimpleNamespace(nodes=(ReplayNode((), ("fast",), "fast", 0, 1, 1),))
        slow = SimpleNamespace(nodes=(ReplayNode((), ("slow",), "slow", 0, 2, 2),))
        result = replay_cohorts(
            fast,
            additional_profiles=(slow,),
            resident_requests=2,
            program_assignment=(1, 1),
            initial_offsets_s=(0, 0.5),
            warmup_rounds=2,
            measured_rounds=4,
        )
        self.assertEqual(result["aggregate_tps"], 0.5)
        self.assertEqual(result["program_assignment"], {0: 1, 1: 1})
        self.assertEqual(result["initial_offsets_s"], [0, 0.5])
        for options in (
            dict(program_assignment=(0,)),
            dict(program_assignment=(0, 2)),
            dict(program_assignment=(True, 0)),
            dict(initial_offsets_s=(0, -1)),
            dict(initial_offsets_s=(0,)),
            dict(initial_offsets_s=(0, 1), initial_spacing_s=1),
        ):
            with self.assertRaises(ConfigurationError):
                replay_cohorts(fast, resident_requests=2, **options)

    def test_hold_includes_propagation_and_matches_serial_arena_replay(self):
        nodes = (
            ReplayNode((), ("arena",), "arena", 7, 0, 0, release_after=3),
            ReplayNode((0,), ("mxm",), "mxm", 0, 2, 2),
            ReplayNode((1,), (), None, 5, 0, 0),
            ReplayNode((2,), ("vxm",), "vxm", 0, 3, 3),
        )
        old = throughput_bounds(nodes, resident_requests=20, batch=1)
        bound = throughput_bounds(
            nodes, resident_requests=20, batch=1, include_lifetimes=True
        )
        self.assertEqual(old["resource_seconds_per_forward"], 3)
        self.assertEqual(bound["resource_seconds_per_forward"], 10)
        self.assertEqual(bound["aggregate_tps"], 0.1)
        kernel = ReplayKernel()
        try:
            profile = kernel.add(nodes)
            kernel.window(0)
            kernel.submit(profile, 0)
            kernel.submit(profile, 0)
            self.assertEqual(kernel.advance().end, 17)
            self.assertEqual(kernel.advance().end, 27)
            self.assertEqual(kernel.stats()["arena"][0], 20)
        finally:
            kernel.close()

    def test_parallel_paths_use_longest_path_and_external_work_is_relaxed(self):
        nodes = (
            ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=4),
            ReplayNode((0,), ("x",), "x", 0, 2, 2),
            ReplayNode((0,), ("y",), "y", 0, 5, 5),
            ReplayNode((), ("external",), "external", 0, 100, 100),
            ReplayNode((1, 2, 3), (), None, 1, 0, 0),
        )
        self.assertEqual(resource_demands(nodes, include_lifetimes=True)["arena"], 6)

    def test_forward_indices_and_external_acquisition_ancestors_are_valid(self):
        nodes = (
            ReplayNode((1,), (), None, 0, 0, 0),
            ReplayNode((3,), ("x",), "x", 0, 2, 2),
            ReplayNode((), ("external",), "external", 0, 100, 100),
            ReplayNode((2,), ("arena",), "arena", 0, 0, 0, release_after=0),
        )
        self.assertEqual(resource_demands(nodes, include_lifetimes=True)["arena"], 2)

    def test_invalid_release_and_duplicate_resource_are_rejected(self):
        nodes = (
            ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=1),
            ReplayNode((), (), None, 0, 2, 0),
        )
        for target in (0, 1, 2):
            with self.assertRaisesRegex(ConfigurationError, "descend"):
                resource_demands(
                    (replace(nodes[0], release_after=target), nodes[1]),
                    include_lifetimes=True,
                )
        with self.assertRaises(ConfigurationError):
            resource_demands((replace(nodes[1], resources=("x", "x")),))
        with self.assertRaises(ConfigurationError):
            resource_demands(nodes, include_lifetimes="yes")


if __name__ == "__main__":
    unittest.main()
