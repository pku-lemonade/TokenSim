"""Shared immutable activation slices retain recipient-specific copy semantics."""

from dataclasses import FrozenInstanceError
import unittest

from TokenSim.af.planning.planner import ParallelExecutionPlanner
from scripts.deepseek_v3_runtime import configuration


class ActivationSliceSharingTest(unittest.TestCase):
    def test_source_slices_are_immutable_and_shared_only_within_one_call(self):
        runtime = configuration(a_tp=2, f_tp=2)
        planner = ParallelExecutionPlanner(runtime.cluster, runtime_plan=runtime.plan)
        bindings = runtime.plan.rank_bindings
        sources = tuple(item for item in bindings if item.domain == "a-domain")
        targets = tuple(item for item in bindings if item.domain == "f-domain")
        tokens = ((3, 8),)
        arguments = dict(
            tokens=tokens,
            a_bindings=sources,
            target_specs=tuple(
                (target, target.physical_identity, tokens) for target in targets
            ),
            producer_ids={
                item.domain_rank: f"attention:{item.domain_rank}" for item in sources
            },
            collective=None,
            barrier_predecessor=None,
            transfer_dtype="fp16",
        )
        copies = planner._activation_copies(**arguments)
        repeated = planner._activation_copies(**arguments)
        self.assertEqual(copies, repeated)
        self.assertEqual(len(copies), 4)
        by_source = {}
        for copy, independent in zip(copies, repeated):
            source = copy.source_binding.domain_rank
            previous = by_source.setdefault(source, copy.tensor_slice)
            self.assertIs(copy.tensor_slice, previous)
            self.assertIsNot(copy.tensor_slice, independent.tensor_slice)
            self.assertEqual(copy.predecessor_id, f"attention:{source}")
            self.assertEqual(copy.consumer_id, copy.target_binding.physical_identity)
            self.assertEqual(copy.tensor_slice.owner_domain_rank, source)
        self.assertEqual(len({id(value) for value in by_source.values()}), 2)
        with self.assertRaises(FrozenInstanceError):
            copies[0].tensor_slice.payload_bytes = 0
