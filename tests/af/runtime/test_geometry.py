"""Af runtime geometry."""

from __future__ import annotations

from dataclasses import replace
import unittest

from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.runtime.records import AFTensorInterval
from TokenSim.errors import ConfigurationError
from tests.af.support.parallel import _cluster
from tests.af.support.runtime import _moe_config, _runtime_hardware, _runtime_plan


class AFRuntimePlanTest(unittest.TestCase):
    def test_ep_modes_reuse_four_f_states_and_resolve_execution_groups(self):
        for ep_size, mode, expert_tp in ((1, "expert_tp", 2), (4, "rank_reuse", 1)):
            with self.subTest(ep_size=ep_size):
                plan = _runtime_plan(
                    _cluster(a_tp=2, a_dp=2, f_tp=2, f_dp=2, f_ep=ep_size)
                )
                f_states = [
                    state
                    for state in plan.physical_states
                    if any(domain == "f-domain" for domain, _ in state.bindings)
                ]
                self.assertEqual(len(f_states), 4)
                self.assertEqual(
                    len({state.physical_identity for state in f_states}), 4
                )

                f_domain = plan.domain_runtime("f-domain")
                self.assertEqual(f_domain.expert_parallel_mode, mode)
                self.assertEqual(f_domain.expert_tensor_parallel_size, expert_tp)
                dense = plan.component_execution("f-domain", 1, "dense_ffn", 1)
                shared = plan.component_execution("f-domain", 0, "shared_expert", 1)
                self.assertEqual(
                    tuple(binding.domain_rank for binding in dense), (2, 3)
                )
                self.assertEqual(
                    tuple(binding.domain_rank for binding in shared), (2, 3)
                )

                routed = plan.expert_execution("f-domain", 3, 1)
                if ep_size == 1:
                    self.assertEqual(
                        tuple(binding.domain_rank for binding in routed), (2, 3)
                    )
                    self.assertTrue(
                        all(binding.ep_group == (2, 3) for binding in routed)
                    )
                    self.assertTrue(all(3 in binding.expert_ids for binding in routed))
                else:
                    self.assertEqual(
                        tuple(binding.domain_rank for binding in routed), (3,)
                    )
                    self.assertEqual(routed[0].ep_group, (0, 1, 2, 3))
                    self.assertEqual(routed[0].ep_rank, 3)

    def test_local_shapes_use_exact_tp_axes_and_reject_ceil_shards(self):
        expert_tp = _runtime_plan(_cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=2, f_ep=1))
        reused = _runtime_plan(_cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=2, f_ep=4))

        self.assertEqual(
            expert_tp.local_shape("f-domain", 2, 0, "routed_expert"),
            {
                "global_hidden_size": 16,
                "hidden_size": 16,
                "tp_size": 2,
                "expert_tp_size": 2,
                "expert_tp_rank": 0,
                "intermediate_size": 16,
            },
        )
        self.assertEqual(
            reused.local_shape("f-domain", 3, 0, "routed_expert"),
            {
                "global_hidden_size": 16,
                "hidden_size": 16,
                "tp_size": 1,
                "expert_tp_size": 1,
                "expert_tp_rank": 0,
                "intermediate_size": 32,
            },
        )
        self.assertEqual(
            expert_tp.local_shape("f-domain", 2, 1, "dense_ffn")["hidden_size"],
            16,
        )
        self.assertEqual(
            expert_tp.local_shape("f-domain", 2, 0, "shared_expert"),
            {
                "global_hidden_size": 16,
                "hidden_size": 16,
                "tp_size": 2,
                "intermediate_size": 16,
            },
        )

        hardware = _runtime_hardware()
        model = replace(
            hardware.model("TestModel"),
            hidden_size=24,
            moe=replace(_moe_config(), hidden_size=24),
            head_dim=3,
        )
        with self.assertRaisesRegex(
            ConfigurationError, r"query heads=8.*A domain='a-domain' TP=3"
        ):
            compile_af_runtime_plan(
                cluster_config=_cluster(a_tp=3, a_dp=1, f_tp=1, f_dp=1, f_ep=1),
                worker_configs=_cluster(
                    a_tp=3, a_dp=1, f_tp=1, f_dp=1, f_ep=1
                ).workers(),
                hardware=hardware,
                model=model,
                block_size=4,
                dtype="fp16",
            )

        with self.assertRaisesRegex(
            ConfigurationError, "Routed Expert intermediate=33"
        ):
            _runtime_plan(
                _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1),
                moe_config=replace(
                    _moe_config(),
                    moe_intermediate_size=33,
                    num_shared_experts=0,
                ),
            )
        with self.assertRaisesRegex(ConfigurationError, "Dense FFN intermediate=65"):
            _runtime_plan(
                _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1),
                moe_config=replace(_moe_config(), intermediate_size=65),
            )
        with self.assertRaisesRegex(ConfigurationError, "Shared FFN intermediate=33"):
            _runtime_plan(
                _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1),
                moe_config=replace(
                    _moe_config(),
                    moe_intermediate_size=33,
                    num_shared_experts=1,
                ),
            )
        rank_reuse_uneven_expert = _runtime_plan(
            _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=2),
            moe_config=replace(
                _moe_config(),
                moe_intermediate_size=33,
                num_shared_experts=0,
            ),
        )
        self.assertEqual(
            rank_reuse_uneven_expert.local_shape("f-domain", 0, 0, "routed_expert")[
                "intermediate_size"
            ],
            33,
        )

    def test_runtime_kv_replication_is_preserved_in_local_shapes(self):
        plan = _runtime_plan(
            _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1),
            moe_config=replace(_moe_config(), num_key_value_heads=1),
        )
        bindings = plan.bindings_for_instance("a-domain", "a0")

        self.assertEqual(
            [binding.kv_partition.global_head_ids for binding in bindings],
            [(0,), (0,)],
        )
        self.assertEqual(
            [binding.kv_partition.replica_factor for binding in bindings],
            [2, 2],
        )
        self.assertEqual(
            [
                plan.local_shape("a-domain", binding.domain_rank, 0, "attention")[
                    "num_kv_heads"
                ]
                for binding in bindings
            ],
            [1, 1],
        )

    def test_tensor_ownership_maps_mismatched_tp_by_global_intersections(self):
        cases = (
            (
                2,
                4,
                ((0, 0, 0, 4), (1, 0, 4, 8), (2, 1, 8, 12), (3, 1, 12, 16)),
            ),
            (
                4,
                2,
                ((0, 0, 0, 4), (0, 1, 4, 8), (1, 2, 8, 12), (1, 3, 12, 16)),
            ),
        )
        for a_tp, f_tp, expected in cases:
            with self.subTest(a_tp=a_tp, f_tp=f_tp):
                plan = _runtime_plan(
                    _cluster(
                        a_tp=a_tp,
                        a_dp=1,
                        f_tp=f_tp,
                        f_dp=1,
                        f_ep=f_tp,
                    )
                )
                a_owners = plan.ownerships_for_replica("a-domain", 0)
                f_owners = plan.ownerships_for_replica("f-domain", 0)
                self.assertEqual(
                    [(item.interval.start, item.interval.end) for item in a_owners],
                    [
                        (rank * (16 // a_tp), (rank + 1) * (16 // a_tp))
                        for rank in range(a_tp)
                    ],
                )
                self.assertEqual(
                    [(item.interval.start, item.interval.end) for item in f_owners],
                    [
                        (rank * (16 // f_tp), (rank + 1) * (16 // f_tp))
                        for rank in range(f_tp)
                    ],
                )
                self.assertTrue(all(item.retains_residual for item in a_owners))
                self.assertTrue(all(not item.retains_residual for item in f_owners))

                destinations = tuple(
                    item
                    for item in plan.partial_destinations
                    if item.phase == "decode"
                    and item.source_dp_rank == 0
                    and item.target_dp_rank == 0
                )
                actual = tuple(
                    (
                        item.source_rank,
                        item.target_rank,
                        item.interval.start,
                        item.interval.end,
                    )
                    for item in destinations
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    sum(item.interval.size for item in destinations),
                    plan.hidden_size,
                )
                self.assertEqual(
                    len(actual),
                    len(set(actual)),
                )

    def test_static_validation_rejects_missing_leader_and_bad_tp_group(self):
        plan = _runtime_plan()
        leader = plan.binding("a-domain", 0)
        no_leader = replace(
            plan,
            rank_bindings=tuple(
                (
                    replace(item, is_scheduler_leader=False)
                    if item.rank_key == leader.rank_key
                    else item
                )
                for item in plan.rank_bindings
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "one scheduler leader"):
            no_leader.validate()

        bad_group = replace(
            plan,
            rank_bindings=tuple(
                (
                    replace(item, tp_group=(item.domain_rank,))
                    if item.rank_key == leader.rank_key
                    else item
                )
                for item in plan.rank_bindings
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "TP group mismatch"):
            bad_group.validate()

        f_binding = plan.binding("f-domain", 0)
        bad_ep_group = replace(
            plan,
            rank_bindings=tuple(
                (
                    replace(item, ep_group=(item.domain_rank,))
                    if item.rank_key == f_binding.rank_key
                    else item
                )
                for item in plan.rank_bindings
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "EP group mismatch"):
            bad_ep_group.validate()

        a_ownership = next(
            item for item in plan.tensor_ownerships if item.domain == "a-domain"
        )
        bad_residual = replace(
            plan,
            tensor_ownerships=tuple(
                replace(item, retains_residual=False) if item == a_ownership else item
                for item in plan.tensor_ownerships
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "retains_residual"):
            bad_residual.validate()

        destination = plan.partial_destinations[0]
        bad_destination = replace(
            plan,
            partial_destinations=(
                replace(destination, interval=AFTensorInterval(0, 1)),
                *plan.partial_destinations[1:],
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "expected"):
            bad_destination.validate()
