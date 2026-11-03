"""LP30 scalar kernels and C2C reductions share the physical vector unit."""

from types import SimpleNamespace
import unittest

from TokenSim.comm.streaming import StreamingConfig
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.streaming import StreamingBuilder


def refined_compute(
    operation, *, event_type="compute", resources=None, hardware="groq_lp30"
):
    worker = "test/worker:0"
    resources = (worker,) if resources is None else resources
    node = ReplayNode((), resources, worker, 0, 1, 1)
    device = SimpleNamespace(analytical={"lpx_fabric": {}})
    runtime = SimpleNamespace(
        hardware=SimpleNamespace(device=lambda name: device),
        workers=(
            SimpleNamespace(
                physical_worker_ref=SimpleNamespace(identity="chip:0", worker_index=0),
                hardware=hardware,
            ),
        ),
    )
    event = SimpleNamespace(
        node_id="scalar",
        operation=operation,
        event_type=event_type,
        physical_endpoints=("chip:0",),
    )
    profile = NativeProfile((node,), {"phase": "decode"})
    return StreamingBuilder(runtime, (event,), profile, StreamingConfig()).nodes[0]


class VectorResourceTest(unittest.TestCase):
    def test_scalar_compute_reserves_the_same_vector_unit_as_collectives(self):
        for operation in (
            "post_attention_norm",
            "residual",
            "data_reorder",
            "embedding_lookup",
            "final_norm",
            "token_selection",
            "dense_ffn",
        ):
            with self.subTest(operation=operation):
                node = refined_compute(operation)
                self.assertIn("lp30:0:vxm", node.resources)
                self.assertEqual((node.duration, node.occupancy), (1, 1))

    def test_scalar_compute_cannot_overlap_a_same_chip_vector_reduction(self):
        node = refined_compute("post_attention_norm")
        reduction = ReplayNode((), ("lp30:0:vxm",), "lp30:0:vxm", 0, 1, 1)
        kernel = ReplayKernel()
        self.addCleanup(kernel.close)
        for item in (node, reduction):
            kernel.submit(kernel.add((item,)), 0)
        self.assertEqual(sorted((kernel.advance().end, kernel.advance().end)), [1, 2])

    def test_existing_vector_binding_is_not_duplicated(self):
        node = refined_compute("attention", resources=("test/worker:0", "lp30:0:vxm"))
        self.assertEqual(node.resources.count("lp30:0:vxm"), 1)

    def test_transport_does_not_acquire_the_vector_unit(self):
        node = refined_compute("af_pipeline_handoff", event_type="communication")
        self.assertEqual(node.resources, ("test/worker:0",))

    def test_other_devices_do_not_acquire_an_lp30_vector_unit(self):
        node = refined_compute("post_attention_norm", hardware="rubin")
        self.assertEqual(node.resources, ("test/worker:0",))


if __name__ == "__main__":
    unittest.main()
