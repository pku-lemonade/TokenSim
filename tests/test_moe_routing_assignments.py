from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe import ExpertRouting, LayerRoutingSnapshot, RoutingAssignment
from TokenSim.moe.config import MoEModelConfig, RoutingConfig


class SyntheticPartitionTest(unittest.TestCase):
    def test_routes_ignore_partition_order_iteration_and_batch_members(self):
        tokens = {request: (7, 8, 11) for request in (1, 9, 23, 41)}
        owners = {request: (0, 0) for request in tokens}

        def routes(router, items, iteration=0):
            return {
                (a.request_id, a.token_id, a.top_k_index): (
                    a.expert_id,
                    a.routing_weight,
                )
                for a in router._synthetic_assignments(
                    items,
                    owners,
                    iteration_id=iteration,
                    phase="decode",
                    decode_step=iteration,
                    layer_id=2,
                )
            }

        for distribution in ("uniform", "hot", "skew", "burst"):
            for fraction in (0.0, 0.75, 1.0):
                with self.subTest(distribution=distribution, fraction=fraction):
                    config = RoutingConfig(
                        distribution=distribution,
                        hot_experts=[0, 1],
                        hot_expert_fraction=fraction,
                    )
                    router = ExpertRouting(_moe(), seed=17, routing_config=config)
                    whole = routes(router, tokens)
                    split = {}
                    for request in reversed(tokens):
                        for token in reversed(tokens[request]):
                            split.update(
                                routes(router, {request: (token,)}, iteration=12)
                            )
                    self.assertEqual(whole, split)
                    self.assertEqual(
                        whole,
                        routes(
                            ExpertRouting(_moe(), seed=17, routing_config=config),
                            tokens,
                        ),
                    )
                    for request, token_ids in tokens.items():
                        for token in token_ids:
                            selected = [whole[(request, token, k)] for k in range(2)]
                            self.assertEqual(len({e for e, _ in selected}), 2)
                            self.assertEqual(sum(w for _, w in selected), 1.0)

    def test_seed_layer_token_and_phase_change_routes_and_hot_pool_saturates(self):
        router = ExpertRouting(_moe(), seed=0)
        baseline = router._experts_for_token(1, 8, "decode", 0)
        for changed in (
            router._experts_for_token(1, 9, "decode", 0),
            router._experts_for_token(1, 8, "prefill", 0),
            router._experts_for_token(1, 8, "decode", 1),
            ExpertRouting(_moe(), seed=1)._experts_for_token(1, 8, "decode", 0),
        ):
            self.assertNotEqual(baseline, changed)
        router = ExpertRouting(
            _moe(top_k=8),
            routing_config=RoutingConfig(
                distribution="hot", hot_experts=[0], hot_expert_fraction=1.0
            ),
        )
        selected = router._experts_for_token(1, 8, "decode", 0)
        self.assertEqual(selected[0], 0)
        self.assertEqual(set(selected), set(range(8)))


def _moe(*, num_experts: int = 8, top_k: int = 2) -> MoEModelConfig:
    return MoEModelConfig(
        is_moe_model=True,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_moe_layers=2,
    )


def _decode_request(request_id: int, *, prefill_len: int = 8) -> Request:
    request = Request(request_id, prefill_len, decode_len=4, block_size=16)
    request.generation_idx = 1
    return request


def _record(
    *,
    step: int,
    layer: int,
    token: int,
    route: int,
    expert: int,
    a_dp_rank: int = 0,
    a_tp_rank: int = 0,
) -> dict[str, object]:
    return {
        "decode_step": step,
        "layer_id": layer,
        "token_id": token,
        "top_k_index": route,
        "expert_id": expert,
        "routing_weight": 0.5,
        "a_dp_rank": a_dp_rank,
        "a_tp_rank": a_tp_rank,
    }


class RoutingAssignmentTest(unittest.TestCase):
    def test_snapshot_is_frozen_and_derives_histogram_from_weighted_edges(self):
        assignments = (
            RoutingAssignment(3, "decode", 1, 4, 7, 12, 0, 2, 0.7, 1, 0),
            RoutingAssignment(3, "decode", 1, 4, 7, 12, 1, 5, 0.3, 1, 0),
        )
        snapshot = LayerRoutingSnapshot(
            iteration_id=3,
            phase="decode",
            decode_step=1,
            layer_id=4,
            top_k=2,
            original_tokens=((7, 12),),
            assignments=assignments,
        )

        self.assertEqual(snapshot.histogram, {2: 1, 5: 1})
        self.assertEqual(snapshot.original_token_count, 1)
        self.assertEqual(snapshot.assignment_count, 2)
        self.assertEqual(
            [
                (item.routing_weight, item.a_dp_rank, item.a_tp_rank)
                for item in assignments
            ],
            [(0.7, 1, 0), (0.3, 1, 0)],
        )
        with self.assertRaises(FrozenInstanceError):
            snapshot.layer_id = 5

    def test_snapshot_canonicalizes_mutable_inputs(self):
        assignments = [
            RoutingAssignment(3, "decode", 1, 4, 7, 12, 0, 2, 0.7, 1, 0),
            RoutingAssignment(3, "decode", 1, 4, 7, 12, 1, 5, 0.3, 1, 0),
        ]
        original_tokens = [[7, 12]]
        snapshot = LayerRoutingSnapshot(
            iteration_id=3,
            phase="decode",
            decode_step=1,
            layer_id=4,
            top_k=2,
            original_tokens=original_tokens,
            assignments=assignments,
        )

        assignments.clear()
        original_tokens[0][1] = 99
        self.assertEqual(snapshot.original_tokens, ((7, 12),))
        self.assertEqual(snapshot.assignment_count, 2)

    def test_balanced_router_covers_experts_and_preserves_owners(self):
        routing = ExpertRouting(_moe())
        requests = [_decode_request(request_id) for request_id in range(4)]

        snapshot = routing.snapshot_for_layer(
            requests,
            iteration_id=9,
            phase="decode",
            decode_step=1,
            layer_id=2,
            a_dp_rank=1,
            a_tp_rank={request.id: request.id % 2 for request in requests},
        )

        self.assertEqual(snapshot.assignment_count, 8)
        self.assertEqual(snapshot.histogram, {expert_id: 1 for expert_id in range(8)})
        self.assertEqual(
            sorted(item.expert_id for item in snapshot.assignments), list(range(8))
        )
        self.assertTrue(all(item.a_dp_rank == 1 for item in snapshot.assignments))
        self.assertEqual(
            [item.a_tp_rank for item in snapshot.assignments],
            [0, 0, 1, 1, 0, 0, 1, 1],
        )

    def test_decode_recompute_routes_every_recomputed_token(self):
        routing = ExpertRouting(_moe(num_experts=8, top_k=2))
        request = _decode_request(7)
        request.prepare_recompute()

        snapshot = routing.snapshot_for_layer(
            [request],
            iteration_id=1,
            phase="decode",
            decode_step=1,
            layer_id=0,
        )

        self.assertEqual(snapshot.original_token_count, request.recompute_tokens)
        self.assertEqual(
            snapshot.assignment_count,
            request.recompute_tokens * routing.moe_config.num_experts_per_tok,
        )

    def test_external_assignments_and_histogram_must_match(self):
        routing = ExpertRouting(_moe())
        request = _decode_request(7)
        request.decode_routing_assignments = [
            _record(step=1, layer=2, token=8, route=0, expert=3),
            _record(step=1, layer=2, token=8, route=1, expert=7),
        ]
        request.decode_expert_histogram = {3: 1, 7: 1}

        snapshot = routing.snapshot_for_layer(
            [request],
            iteration_id=1,
            phase="decode",
            decode_step=1,
            layer_id=2,
        )
        self.assertEqual(snapshot.histogram, {3: 1, 7: 1})

        request.decode_expert_histogram = {3: 2}
        with self.assertRaisesRegex(WorkloadValidationError, "histogram mismatch"):
            routing.snapshot_for_layer(
                [request],
                iteration_id=2,
                phase="decode",
                decode_step=1,
                layer_id=2,
            )

    def test_external_assignment_requires_runtime_owner(self):
        routing = ExpertRouting(_moe())
        request = _decode_request(7)
        first = _record(step=1, layer=2, token=8, route=0, expert=3)
        first.pop("a_tp_rank")
        request.decode_routing_assignments = [
            first,
            _record(step=1, layer=2, token=8, route=1, expert=7),
        ]

        with self.assertRaisesRegex(WorkloadValidationError, "missing a_tp_rank owner"):
            routing.snapshot_for_layer(
                [request],
                iteration_id=1,
                phase="decode",
                decode_step=1,
                layer_id=2,
            )

    def test_histogram_only_remains_statistics_but_cannot_drive_snapshot(self):
        routing = ExpertRouting(_moe())
        request = _decode_request(7)
        request.decode_expert_histogram = {3: 1, 7: 1}

        self.assertEqual(
            routing.histogram_for_step(
                [request], phase="decode", decode_step=1, layer_id=2
            ),
            {3: 1, 7: 1},
        )
        with self.assertRaisesRegex(WorkloadValidationError, "statistics-only"):
            routing.snapshot_for_layer(
                [request],
                iteration_id=1,
                phase="decode",
                decode_step=1,
                layer_id=2,
            )

    def test_legacy_prefill_histogram_does_not_require_decode_step(self):
        routing = ExpertRouting(_moe(num_experts=4, top_k=1))
        request = Request(7, prefill_len=2, decode_len=1, block_size=16)
        request.prefill_expert_histogram = {0: 1, 2: 1}

        self.assertEqual(
            routing.histogram_for_step([request], phase="prefill"),
            {0: 1, 2: 1},
        )

    def test_external_records_are_isolated_by_step_and_layer(self):
        routing = ExpertRouting(_moe())
        request = _decode_request(7)
        request.decode_routing_assignments = [
            _record(step=1, layer=0, token=8, route=0, expert=0),
            _record(step=1, layer=0, token=8, route=1, expert=1),
            _record(step=1, layer=1, token=8, route=0, expert=2),
            _record(step=1, layer=1, token=8, route=1, expert=3),
            _record(step=2, layer=0, token=9, route=0, expert=4),
            _record(step=2, layer=0, token=9, route=1, expert=5),
        ]

        layer_zero = routing.snapshot_for_layer(
            [request],
            iteration_id=1,
            phase="decode",
            decode_step=1,
            layer_id=0,
        )
        layer_one = routing.snapshot_for_layer(
            [request],
            iteration_id=1,
            phase="decode",
            decode_step=1,
            layer_id=1,
        )
        request.generation_idx = 2
        next_step = routing.snapshot_for_layer(
            [request],
            iteration_id=2,
            phase="decode",
            decode_step=2,
            layer_id=0,
        )

        self.assertEqual(layer_zero.histogram, {0: 1, 1: 1})
        self.assertEqual(layer_one.histogram, {2: 1, 3: 1})
        self.assertEqual(next_step.histogram, {4: 1, 5: 1})

    def test_legacy_histogram_cache_is_scoped_by_step_and_layer(self):
        routing = ExpertRouting(_moe())
        request = _decode_request(7)
        request.decode_expert_histogram = {0: 1, 1: 1}
        self.assertEqual(
            routing.histogram_for_step(
                [request], phase="decode", decode_step=1, layer_id=0
            ),
            {0: 1, 1: 1},
        )
        request.decode_expert_histogram = {2: 1, 3: 1}
        self.assertEqual(
            routing.histogram_for_step(
                [request], phase="decode", decode_step=1, layer_id=1
            ),
            {2: 1, 3: 1},
        )
        request.decode_expert_histogram = {4: 1, 5: 1}
        self.assertEqual(
            routing.histogram_for_step(
                [request], phase="decode", decode_step=2, layer_id=0
            ),
            {4: 1, 5: 1},
        )


if __name__ == "__main__":
    unittest.main()
