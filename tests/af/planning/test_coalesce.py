"""Ready transfer batches retain bytes, tensor slices and producer dependencies."""

from dataclasses import replace
import unittest

from TokenSim.af.planning.coalesce import coalesce_ready_transfers, rewrite_dependencies
from TokenSim.af.planning.nodes import ExecutionNode
from TokenSim.af.providers.queries import AFTensorSlice


class CoalesceTest(unittest.TestCase):
    def test_only_equivalent_readiness_and_endpoints_are_batched(self):
        first = ExecutionNode(
            node_id="one",
            operation="af_f_to_a",
            kind="communication",
            predecessor_ids=("producer",),
            stage=None,
            direction="f_to_a",
            source_domain="f",
            target_domain="a",
            logical_participants=("f", "a"),
            domain_ranks=(0, 1),
            physical_workers=("f:0", "a:0"),
            dp_rank=0,
            network="net",
            payload_bytes=32,
            tensor_slices=(
                AFTensorSlice("first", (0, 16), "partial", payload_bytes=32),
            ),
        )
        second = replace(
            first,
            node_id="two",
            payload_bytes=64,
            tensor_slices=(
                AFTensorSlice("second", (0, 32), "partial", payload_bytes=64),
            ),
        )
        different = replace(first, node_id="three", predecessor_ids=("later",))
        nodes = coalesce_ready_transfers((first, second, different))
        self.assertEqual(len(nodes), 2)
        self.assertEqual(sum(node.payload_bytes for node in nodes), 128)
        self.assertEqual(nodes[0].predecessor_ids, ("producer",))
        self.assertEqual(sum(item.payload_bytes for item in nodes[0].tensor_slices), 96)
        self.assertIs(nodes[1], different)

    def test_aliases_preserve_endpoint_order_and_deduplicate_dependencies(self):
        node = ExecutionNode(
            node_id="consumer",
            operation="all_reduce",
            kind="communication",
            predecessor_ids=("left", "right"),
            stage=None,
            direction=None,
            source_domain="a",
            target_domain="a",
            logical_participants=("a:0", "a:1"),
            domain_ranks=(0, 1),
            physical_workers=("worker:0", "worker:1"),
            dp_rank=0,
            input_producer_ids=(("worker:0", "left"), ("worker:1", "right")),
        )
        self.assertIs(rewrite_dependencies((node,), {})[0], node)
        rewritten = rewrite_dependencies(
            (node,), {"left": "joined", "right": "joined"}
        )[0]
        self.assertEqual(rewritten.predecessor_ids, ("joined",))
        self.assertEqual(
            rewritten.input_producer_ids,
            (("worker:0", "joined"), ("worker:1", "joined")),
        )
        self.assertEqual(node.predecessor_ids, ("left", "right"))
        duplicate = replace(node, predecessor_ids=("left", "right", "left"))
        self.assertEqual(
            rewrite_dependencies((duplicate,), {})[0].predecessor_ids, ("left", "right")
        )
