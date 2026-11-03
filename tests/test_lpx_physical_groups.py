"""Independent LP30 worker groups retain distinct physical fabric resources."""

from types import SimpleNamespace
import unittest

from TokenSim.comm.streaming import StreamingConfig
from TokenSim.config.af.domains import PhysicalWorkerRef
from TokenSim.errors import ConfigurationError
from scripts.lpx.fabric import LP30Fabric
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.streaming import StreamingBuilder


def worker(group, index, hardware="groq_lp30"):
    return SimpleNamespace(
        physical_worker_ref=PhysicalWorkerRef(group, index), hardware=hardware
    )


def builder(workers, *, compute=False):
    device = SimpleNamespace(analytical={"lpx_fabric": {}})
    runtime = SimpleNamespace(
        hardware=SimpleNamespace(device=lambda name: device), workers=workers
    )
    nodes, events = [], []
    if compute:
        for i, item in enumerate(workers):
            identity = item.physical_worker_ref.identity
            resource = f"test/worker:{identity}"
            nodes.append(ReplayNode((), (resource,), resource, 0, 1, 1))
            events.append(
                SimpleNamespace(
                    node_id=str(i),
                    operation="post_attention_norm",
                    event_type="compute",
                    physical_endpoints=(identity,),
                )
            )
    return StreamingBuilder(
        runtime,
        tuple(events),
        NativeProfile(tuple(nodes), {"phase": "decode"}),
        StreamingConfig(),
    )


class PhysicalGroupTest(unittest.TestCase):
    def test_single_group_preserves_native_indices_and_scope(self):
        b = builder((worker("a", 15), worker("a", 16)))
        self.assertEqual(b.worker_indices, {"a:15": 15, "a:16": 16})
        self.assertEqual(b.fabric.chips(("a:15", "a:16")), (15, 16))
        self.assertAlmostEqual(b.fabric.policy.propagation_s((15, 16)), 0.75e-6)

    def test_groups_receive_disjoint_coordinate_ranges(self):
        b = builder(tuple(worker(group, i) for group in ("a", "f") for i in range(2)))
        self.assertEqual(b.worker_indices, {"a:0": 0, "a:1": 1, "f:0": 2, "f:1": 3})
        self.assertEqual(b.fabric.chips(("a:0", "f:0")), (0, 2))
        resources, _, _ = b.fabric.collective_phase(("f:0", "f:1"), 320)
        self.assertIn("lp30:2:tx", resources)
        self.assertNotIn("lp30:0:tx", resources)

    def test_compute_on_distinct_group_chips_can_overlap(self):
        b = builder((worker("a", 0), worker("f", 0)), compute=True)
        self.assertIn("lp30:0:vxm", b.nodes[0].resources)
        self.assertIn("lp30:1:vxm", b.nodes[1].resources)
        kernel = ReplayKernel()
        self.addCleanup(kernel.close)
        for node in b.nodes:
            kernel.submit(kernel.add((node,)), 0)
        self.assertEqual([kernel.advance().end, kernel.advance().end], [1, 1])

    def test_sparse_groups_do_not_alias_or_compress_native_indices(self):
        b = builder((worker("a", 3), worker("a", 7), worker("f", 2)))
        self.assertEqual(b.worker_indices, {"a:3": 3, "a:7": 7, "f:2": 10})

    def test_other_hardware_does_not_reserve_lp30_coordinates(self):
        b = builder((worker("gpu", 99, "rubin"), worker("a", 0), worker("f", 0)))
        self.assertEqual(b.worker_indices, {"a:0": 0, "f:0": 1})

    def test_routes_cannot_borrow_a_non_lp30_worker_index(self):
        b = builder((worker("a", 0), worker("a", 1), worker("gpu", 99, "rubin")))
        b.fabric.policy = LP30Fabric(routes=({"source": 0, "target": 1, "via": [99]},))
        with self.assertRaisesRegex(ConfigurationError, "allocated physical"):
            b.fabric.routed_phase([(0, 1, 320)], 6)


if __name__ == "__main__":
    unittest.main()
