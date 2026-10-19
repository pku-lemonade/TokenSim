"""Structural keys and immutable wire plans retain complete service inputs."""

from dataclasses import dataclass, replace
import gc
import unittest
import weakref

from TokenSim.af.caching.estimates import (
    FingerprintMemo,
    FrozenDict,
    identity,
    prepare_query,
    query_skeleton,
    snapshot,
)
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.af.providers.queries import AFComputeQuery, AFRoutingEdge, AFTensorSlice
from TokenSim.errors import ConfigurationError
from scripts.deepseek_v3_runtime import configuration
from tests.af.providers import test_cache
from tests.af.support.model import _compute_query, _transfer_query


@dataclass(frozen=True)
class FrozenRecord:
    value: object


@dataclass
class MutableRecord:
    value: object


class StructuralKeyTest(unittest.TestCase):
    def test_keys_match_reference_for_nested_mutable_and_immutable_values(self):
        values = [
            None,
            True,
            1,
            1.0,
            (),
            ("worker", 2, None),
            ((1, True, 1.0), [2, False]),
            {"b": [1, True], "a": {"float": 2.0}},
            FrozenDict({"tuple": (2, None), "record": FrozenRecord(1.0)}),
            FrozenRecord({"mutable": [1, 2]}),
            MutableRecord((1, True)),
            snapshot(MutableRecord({"frozen": [1, 2.0]})),
        ]
        memo = FingerprintMemo(capacity=4)
        for _ in range(2):
            for value in values:
                with self.subTest(value=value):
                    self.assertEqual(memo(value), identity(value))
                    self.assertLessEqual(len(memo.values), 4)
        self.assertEqual(len({memo((item,)) for item in (True, 1, 1.0)}), 3)

    def test_frozen_parent_with_mutable_descendants_is_revisited(self):
        child = {"shape": [1, 2]}
        parent = FrozenRecord((child,))
        memo = FingerprintMemo()
        original = memo(parent)
        child["shape"][0] = True
        self.assertNotEqual(memo(parent), original)
        self.assertEqual(memo(parent), identity(parent))
        self.assertNotIn(id(parent), memo.values)
        self.assertNotIn(id(parent.value), memo.values)

    def test_extraction_preserves_defaults_and_execution_field_exclusions(self):
        query = _compute_query()
        skeleton = query_skeleton(AFComputeQuery)
        values = {
            name: value for name, value in vars(query).items() if value is not None
        }
        expected = tuple(
            values.get(name, skeleton.defaults.get(name)) for name in skeleton.names
        )
        before = values.copy()
        self.assertEqual(skeleton.key_values(values), expected)
        self.assertEqual(values, before)
        self.assertNotIn("iteration_id", skeleton.names)
        self.assertNotIn("routing_edges", skeleton.names)
        del values["token_count"]
        self.assertEqual(
            skeleton.key_values(values),
            tuple(
                values.get(name, skeleton.defaults.get(name)) for name in skeleton.names
            ),
        )


class WirePlanKeyTest(unittest.TestCase):
    def provider(self):
        return test_cache.CacheFastPathTest().transfer()

    def test_wire_geometry_shares_plans_without_retaining_invocation_records(self):
        provider = self.provider()
        base = _transfer_query()
        plan = provider.communication_plan(base)
        edge = AFRoutingEdge(3, "decode", 3, 0, 1, 7, 0, 2, 1.0, 0, 0)
        tensor = AFTensorSlice("request:1:token:7:activation", (0, 4), "sharded")
        edge_ref, tensor_ref = weakref.ref(edge), weakref.ref(tensor)
        changed = replace(
            base,
            iteration_id=3,
            routing_edges=(edge,),
            tensor_slices=(tensor,),
            topology={"invocation_note": "updated"},
            logical_input_bytes_per_rank=((0, 123),),
        )
        prepared = prepare_query(provider, type(changed), **vars(changed))
        self.assertIs(provider.communication_plan(prepared), plan)
        self.assertEqual(provider.plan_cache_hits, 1)
        self.assertEqual(prepared.routing_edges, (edge,))
        self.assertEqual(prepared.tensor_slices, (tensor,))
        del changed, prepared, edge, tensor
        gc.collect()
        self.assertIsNone(edge_ref())
        self.assertIsNone(tensor_ref())

    def test_geometry_and_backend_changes_cannot_reuse_wrong_plan(self):
        provider = self.provider()
        base = _transfer_query()
        first = provider.communication_plan(base)
        for changed in (
            replace(base, payload_bytes=8192),
            replace(base, dtype="bf16"),
            replace(base, physical_endpoints=tuple(reversed(base.physical_endpoints))),
            replace(base, transport_backend="pynccl"),
        ):
            with self.subTest(query=changed):
                self.assertIsNot(provider.communication_plan(changed), first)
        backend = provider.communication_plan(replace(base, transport_backend="pynccl"))
        self.assertTrue(
            all(phase.transport_backend == "pynccl" for phase in backend.phases)
        )

    def test_invalid_collective_still_fails_after_plan_cache_is_warm(self):
        provider = self.provider()
        query = replace(_transfer_query(), operation="all_reduce", group_size=2)
        provider.communication_plan(query)
        for changed in (
            replace(query, group_size=3),
            replace(query, participant_ranks=(0, 0)),
            replace(query, participant_ranks=(0,)),
            replace(query, transport_backend=""),
        ):
            with self.subTest(query=changed), self.assertRaises(ConfigurationError):
                provider.communication_plan(changed)

    def test_endpoint_interning_is_scoped_to_one_planner_and_full_geometry(self):
        config = configuration(a_tp=2, f_tp=2)
        planners = [
            ParallelExecutionPlanner(config.cluster, runtime_plan=config.plan)
            for _ in range(2)
        ]
        binding = config.plan.rank_bindings[0]
        first = planners[0]._endpoint(binding)
        self.assertIs(planners[0]._endpoint(binding), first)
        self.assertIsNot(planners[1]._endpoint(binding), first)
        changed = replace(binding, tp_rank=binding.tp_rank + 1)
        self.assertNotEqual(planners[0]._endpoint(changed), first)
        self.assertEqual(len(planners[0]._endpoints), 2)
