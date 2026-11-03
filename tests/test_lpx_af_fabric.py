"""Pure-LP30 A/F transfers share the routed fabric's physical resources."""

from types import SimpleNamespace
import unittest

from TokenSim.errors import ConfigurationError
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.streaming import duration
from tests.test_lpx_physical_groups import builder, worker


def transfer_builder(operation="af_a_to_f", *, endpoints=None, workers=None, **details):
    b = builder(workers or (worker("a", 0), worker("f", 0)))
    original = ReplayNode((), ("native-endpoint",), "native-endpoint", 0, 99, 99)
    b.nodes = [original]
    event = SimpleNamespace(
        operation=operation,
        physical_endpoints=endpoints or ("a:0", "f:0"),
        payload_bytes=320,
        details=details,
        match_type="analytical",
    )
    return b, event


class AFFabricTest(unittest.TestCase):
    def test_both_directions_use_distinct_chip_ports_and_scoped_flight(self):
        for operation, endpoints, source, target in (
            ("af_a_to_f", ("a:0", "f:0"), 0, 1),
            ("af_f_to_a", ("f:0", "a:0"), 1, 0),
        ):
            with self.subTest(operation=operation):
                b, event = transfer_builder(operation, endpoints=endpoints)
                b.fabric.point_to_point(0, event)
                resources, wire, flight = b.fabric.policy.phase(
                    [(source, target, 320)], b.fabric.policy.p2p_lanes
                )
                self.assertEqual(b.nodes[0].resources, ())
                self.assertEqual(b.nodes[1].resources, resources)
                self.assertIn(f"lp30:{source}:tx", resources)
                self.assertIn(f"lp30:{target}:rx", resources)
                self.assertAlmostEqual(duration(b.nodes), wire + flight)
                self.assertEqual(b.fabric.counts["af_transfers"], 1)

    def test_af_and_other_c2c_wire_work_contend_for_endpoint_injection(self):
        b, event = transfer_builder(
            workers=(worker("a", 0), worker("a", 1), worker("f", 0))
        )
        b.fabric.point_to_point(0, event)
        af = b.nodes[1].resources
        other, _, _ = b.fabric.collective_phase(("a:0", "a:1"), 320)
        self.assertIn("lp30:0:tx", set(af) & set(other))
        kernel = ReplayKernel()
        self.addCleanup(kernel.close)
        for resources in (af, other):
            node = ReplayNode((), resources, resources[0], 0, 1, 1)
            kernel.submit(kernel.add((node,)), 0)
        self.assertEqual([kernel.advance().end, kernel.advance().end], [1, 2])

    def test_mixed_hardware_transport_retains_its_native_model(self):
        b, event = transfer_builder(
            endpoints=("a:0", "gpu:0"),
            workers=(worker("a", 0), worker("gpu", 0, "rubin")),
        )
        original = tuple(b.nodes)
        b.fabric.point_to_point(0, event)
        self.assertEqual(tuple(b.nodes), original)

    def test_same_chip_copy_retains_its_native_model(self):
        b, event = transfer_builder(endpoints=("a:0", "a:0"))
        original = tuple(b.nodes)
        b.fabric.point_to_point(0, event)
        self.assertEqual(tuple(b.nodes), original)

    def test_completion_only_phase_is_not_silently_rerouted(self):
        b, event = transfer_builder(phase_completion_only=True)
        with self.assertRaisesRegex(ConfigurationError, "phase"):
            b.fabric.point_to_point(0, event)

    def test_measured_af_transport_is_not_repriced(self):
        b, event = transfer_builder()
        event.match_type = "measured"
        with self.assertRaisesRegex(ConfigurationError, "analytical"):
            b.fabric.point_to_point(0, event)


if __name__ == "__main__":
    unittest.main()
