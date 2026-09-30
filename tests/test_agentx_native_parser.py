from __future__ import annotations

import ast
import json
import math
import random
import unittest
from pathlib import Path

from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.agentx_native.models import (
    Conversation,
    GraphTurn,
    RawRequest,
    SourceKey,
    TurnKey,
)
from TokenSim.workload.agentx_native.parser import (
    infer_predecessors,
    iter_trace_graphs,
    parse_trace_graph,
)
from TokenSim.workload.agentx_native.trajectory import (
    SequentialTracePool,
    cache_scope,
    initial_snapshots,
    primer_schedule,
    sample_snapshot,
    snapshot_at,
)


def row(t, hashes, *, model="model", osl=2, api=0.1, isl=None):
    return {
        "t": t,
        "type": "s",
        "in": isl if isl is not None else len(hashes) * 64,
        "out": osl,
        "hash_ids": hashes,
        "model": model,
        "api_time": api,
    }


def trace(rows):
    return {"id": "trace", "block_size": 64, "hash_id_scope": "local", "requests": rows}


def conversation(sid, requests, *, parent="trace"):
    return Conversation(
        sid,
        None if sid == "trace" else parent,
        0 if sid == "trace" else 1,
        [
            GraphTurn(TurnKey(sid, i), SourceKey("trace", i), req, "weka_main", 0)
            for i, req in enumerate(requests)
        ],
    )


def req(t, duration):
    return RawRequest(t, 64, 1, [1], "model", duration)


class NativeChainParserTest(unittest.TestCase):
    def test_same_prefix_temporal_overlap_creates_independent_parallel_flows(self):
        graph = parse_trace_graph(
            trace(
                [
                    row(0, [1, 2], api=1),
                    row(0.1, [1, 2, 3], api=1),
                    row(2, [1, 2, 4]),
                    row(2.1, [1, 2, 3, 5]),
                ]
            )
        )
        self.assertEqual(graph.request_count, 4)
        self.assertEqual(len(graph.conversations), 2)
        self.assertEqual([t.source.outer_index for t in graph.root.turns], [0, 2])
        self.assertEqual(
            [t.source.outer_index for t in graph.conversations[1].turns], [1, 3]
        )

    def test_compaction_seam_stays_one_session_when_no_future_pullback(self):
        graph = parse_trace_graph(
            trace([row(0, [1, 2, 3, 4]), row(1, [1, 2]), row(2, [1, 2, 5])])
        )
        self.assertEqual(len(graph.conversations), 1)
        self.assertEqual(graph.request_count, 3)

    def test_same_prefix_cross_model_requests_do_not_continue_same_session(self):
        graph = parse_trace_graph(
            trace([row(0, [1, 2]), row(1, [1, 2, 3], model="other"), row(2, [1, 2, 4])])
        )
        self.assertEqual(len(graph.conversations), 2)
        self.assertEqual(graph.conversations[1].conversation_id, "trace::aux:000")
        self.assertTrue(graph.conversations[1].is_auxiliary)

    def test_leading_title_preamble_keeps_original_identity_on_main(self):
        graph = parse_trace_graph(
            trace(
                [
                    row(0, [99], model="title", osl=20),
                    row(0.2, [1, 2]),
                    row(1, [1, 2, 3]),
                ]
            )
        )
        self.assertEqual(len(graph.conversations), 1)
        self.assertEqual([t.source.outer_index for t in graph.root.turns], [0, 1, 2])
        self.assertEqual(graph.root.turns[0].request.model, "title")

    def test_empty_hash_request_is_preserved_on_main(self):
        graph = parse_trace_graph(
            trace([row(0, [], isl=5), row(1, [1, 2]), row(2, [1, 2, 3])])
        )
        self.assertEqual(graph.request_count, 3)
        self.assertEqual(graph.root.turns[0].request.input_length, 5)

    def test_uint64_hash_identity_survives_chain_comparison(self):
        maximum = 2**64 - 1
        graph = parse_trace_graph(trace([row(0, [maximum]), row(1, [maximum, 3])]))
        self.assertEqual(len(graph.conversations), 1)
        self.assertEqual(graph.root.turns[0].request.hash_ids[0], maximum)
        self.assertEqual(graph.root.turns[1].request.hash_ids[0], maximum)

    def test_subagent_relative_time_inner_index_and_sibling_identity_are_preserved(
        self,
    ):
        marker = {
            "type": "subagent",
            "agent_id": "child",
            "t": 10,
            "duration_ms": 4000,
            "requests": [
                row(0, [1, 2], api=1),
                row(0.1, [9], model="other", api=0.2),
                row(2, [1, 2, 3]),
            ],
        }
        graph = parse_trace_graph(trace([row(0, [1]), marker, row(15, [1, 3])]))
        children = graph.conversations[1:]
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0].conversation_id, "trace::sa:child")
        self.assertEqual(children[1].conversation_id, "trace::sa:child:aux:000")
        self.assertEqual([t.source.inner_index for t in children[0].turns], [0, 2])
        self.assertEqual(children[1].turns[0].source.inner_index, 1)
        self.assertEqual(children[1].turns[0].request.t, 10.1)
        self.assertTrue(all(c.parent_conversation_id == "trace" for c in children))
        self.assertEqual(graph.branches[0].join_turn_index, 1)

    def test_empty_subagent_and_background_branch_are_kept(self):
        graph = parse_trace_graph(
            trace(
                [
                    row(0, [1]),
                    {
                        "type": "subagent",
                        "agent_id": "empty",
                        "t": 1,
                        "duration_ms": 20000,
                        "requests": [],
                    },
                ]
            )
        )
        self.assertEqual(graph.request_count, 1)
        self.assertEqual(len(graph.conversations), 2)
        self.assertEqual(graph.conversations[1].turns, [])
        self.assertIsNone(graph.branches[0].join_turn_index)
        self.assertEqual(graph.recorded_activity_end_s, 21)
        self.assertEqual(graph.time_end_s, 0)  # sampling uses request starts

    def test_tiered_subagents_have_distinct_join_targets(self):
        def marker(name, t, duration):
            return {
                "type": "subagent",
                "agent_id": name,
                "t": t,
                "duration_ms": duration,
                "requests": [],
            }

        graph = parse_trace_graph(
            trace(
                [
                    row(0, [1]),
                    marker("short", 1, 2000),
                    marker("long", 1, 12000),
                    row(5, [1, 2]),
                    row(20, [1, 2, 3]),
                ]
            )
        )
        joins = {b.child_conversation_ids[0]: b.join_turn_index for b in graph.branches}
        self.assertEqual(joins, {"trace::sa:short": 1, "trace::sa:long": 2})

    def test_zero_output_is_recorded_and_runtime_minimum_is_one(self):
        graph = parse_trace_graph(trace([row(0, [1], osl=0)]))
        self.assertEqual(graph.root.turns[0].request.output_length, 0)
        self.assertEqual(graph.root.turns[0].requested_output_tokens, 1)

    def test_wait_subtracts_historical_api_time_without_adding_it_to_service(self):
        graph = parse_trace_graph(trace([row(0, [1], api=2), row(5, [1, 2])]))
        self.assertEqual(graph.root.turns[1].wait_after_previous_s, 3)
        self.assertEqual(graph.root.turns[1].request.t, 5)
        # The fixed source version's wait helper uses finite api_time as-is.
        negative = parse_trace_graph(trace([row(0, [1], api=-2), row(5, [1, 2])]))
        self.assertEqual(negative.root.turns[1].wait_after_previous_s, 7)
        self.assertEqual(negative.root.turns[0].request.duration_s, 0)

    def test_sampling_bounds_cover_children_and_use_starts_not_ends(self):
        marker = {
            "type": "subagent",
            "agent_id": "child",
            "t": 1,
            "duration_ms": 100000,
            "requests": [row(20, [4], api=50), row(90, [4, 5], api=20)],
        }
        graph = parse_trace_graph(trace([row(0, [1]), marker, row(2, [1, 2])]))
        self.assertEqual(graph.time_start_s, 0)
        self.assertEqual(graph.time_end_s, 90)
        self.assertEqual(graph.recorded_activity_end_s, 110)

    def test_duplicate_agents_or_invalid_hash_scope_are_rejected(self):
        marker = {"type": "subagent", "agent_id": "child", "t": 1, "requests": []}
        with self.assertRaises(WorkloadValidationError):
            parse_trace_graph(trace([row(0, [1]), marker, marker]))
        raw = trace([row(0, [1])])
        raw["hash_id_scope"] = "global"
        with self.assertRaises(WorkloadValidationError):
            parse_trace_graph(raw)
        with self.assertRaises(WorkloadValidationError):
            parse_trace_graph(trace([row(0, [-1])]))

    def test_repository_fixtures_preserve_every_original_request(self):
        graphs = list(iter_trace_graphs("tests/fixtures/agentx_weka.jsonl"))
        self.assertEqual([g.request_count for g in graphs], [5, 1])
        self.assertEqual([g.raw_request_count for g in graphs], [5, 1])
        self.assertEqual(
            sum(len(set(t.source for t in g.all_turns())) for g in graphs), 6
        )

    def test_native_modules_have_no_aiperf_package_imports(self):
        for path in Path("TokenSim/workload/agentx_native").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    self.assertTrue(
                        all(not a.name.startswith("aiperf") for a in node.names), path
                    )
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("aiperf"), path)


class IntervalPredecessorTest(unittest.TestCase):
    def test_equal_start_zero_duration_is_unordered(self):
        streams = [
            conversation("trace", [req(1, 0)]),
            conversation("trace::fa:000", [req(1, 0)]),
        ]
        self.assertTrue(all(not p for p in infer_predecessors(streams).values()))

    def test_exact_boundary_uses_reference_millisecond_arithmetic(self):
        streams = [
            conversation("trace", [req(0.1, 0.2)]),
            conversation("trace::fa:000", [req(0.3, 1)]),
        ]
        self.assertEqual(
            infer_predecessors(streams)[TurnKey("trace::fa:000", 0)],
            (TurnKey("trace", 0),),
        )

    def test_overlap_has_no_completion_edge_but_later_request_joins(self):
        streams = [
            conversation("trace", [req(0, 10)]),
            conversation("trace::fa:000", [req(2, 1), req(10, 1)]),
        ]
        result = infer_predecessors(streams)
        self.assertEqual(result[TurnKey("trace::fa:000", 0)], ())
        self.assertEqual(result[TurnKey("trace::fa:000", 1)], (TurnKey("trace", 0),))

    def test_transitive_frontier_keeps_only_latest_causal_completion(self):
        streams = [
            conversation("trace", [req(0, 1)]),
            conversation("trace::fa:000", [req(2, 1)]),
            conversation("trace::fa:001", [req(4, 1)]),
        ]
        result = infer_predecessors(streams)
        self.assertEqual(
            result[TurnKey("trace::fa:001", 0)], (TurnKey("trace::fa:000", 0),)
        )

    def test_tagged_subagent_scopes_are_independent_from_parent_frontier(self):
        streams = [
            conversation("trace", [req(0, 1)]),
            conversation("trace::sa:child", [req(2, 1)]),
            conversation("trace::sa:child:fa:000", [req(4, 1)]),
            conversation("trace::sa:other", [req(5, 1)]),
        ]
        result = infer_predecessors(streams)
        self.assertEqual(result[TurnKey("trace::sa:child", 0)], ())
        self.assertEqual(
            result[TurnKey("trace::sa:child:fa:000", 0)],
            (TurnKey("trace::sa:child", 0),),
        )
        self.assertEqual(result[TurnKey("trace::sa:other", 0)], ())

    def test_invalid_duration_is_zero_width_and_latest_tie_uses_turn_key(self):
        streams = [
            conversation("trace", [req(0, math.nan), req(0, -1)]),
            conversation("trace::fa:000", [req(1, 1)]),
        ]
        result = infer_predecessors(streams)
        self.assertEqual(result[TurnKey("trace::fa:000", 0)], (TurnKey("trace", 1),))

    def test_indexed_algorithm_matches_direct_reference_rule_across_varied_intervals(
        self,
    ):
        rng = random.Random(37)
        streams = [
            conversation(
                "trace" if c == 0 else f"trace::fa:{c:03d}",
                [
                    req(rng.randrange(20) / 10, rng.choice([0, 0.1, 0.2, 1, -1, None]))
                    for _ in range(25)
                ],
            )
            for c in range(8)
        ]
        expected = {}
        for stream in streams:
            for target in stream.turns:
                start = target.request.t * 1000
                frontier = []
                for other in streams:
                    if other.conversation_id == stream.conversation_id:
                        continue
                    completed = [
                        (
                            t,
                            t.request.t * 1000,
                            t.request.t * 1000 + t.request.duration_s * 1000,
                        )
                        for t in other.turns
                        if t.request.t * 1000 < start
                        and t.request.t * 1000 + t.request.duration_s * 1000 <= start
                    ]
                    if completed:
                        frontier.append(
                            max(completed, key=lambda x: (x[2], x[1], x[0].key))
                        )
                expected[target.key] = tuple(
                    sorted(
                        c[0].key
                        for c in frontier
                        if not any(
                            c[1] < later[1] and c[2] <= later[1]
                            for later in frontier
                            if later is not c
                        )
                    )
                )
        self.assertEqual(infer_predecessors(streams), expected)


class TrajectoryPlanTest(unittest.TestCase):
    def graph(self):
        return parse_trace_graph(
            trace(
                [
                    row(0, [1]),
                    {
                        "type": "subagent",
                        "agent_id": "child",
                        "t": 1,
                        "duration_ms": 5000,
                        "requests": [row(1, [1, 4]), row(5, [1, 4, 5])],
                    },
                    row(7, [1, 2]),
                ]
            )
        )

    def test_waiting_parent_is_kept_and_primed_with_its_live_child(self):
        snap = snapshot_at(self.graph(), 0, 4000)
        self.assertEqual(len(snap.states), 2)
        states = {s.conversation_id: s for s in snap.states}
        self.assertTrue(states["trace"].waiting_on_children)
        self.assertEqual(states["trace"].next_turn_index, 1)
        self.assertEqual(states["trace"].primer_turn_index, 0)
        self.assertFalse(states["trace::sa:child"].waiting_on_children)
        self.assertEqual(
            states["trace::sa:child"].join_memberships, (("trace:spawn:child", 1),)
        )
        self.assertEqual(len(primer_schedule([snap])), 2)

    def test_root_finished_but_background_child_keeps_snapshot_live(self):
        graph = parse_trace_graph(
            trace(
                [
                    row(0, [1]),
                    {
                        "type": "subagent",
                        "agent_id": "child",
                        "t": 1,
                        "requests": [row(1, [4]), row(9, [4, 5])],
                    },
                    row(2, [1, 2]),
                ]
            )
        )
        snap = snapshot_at(graph, 0, 6000)
        self.assertEqual([s.conversation_id for s in snap.states], ["trace::sa:child"])
        self.assertEqual(dict(snap.completed_boundaries)["trace"], 2)
        self.assertEqual(snap.root_next_turn_index, 0)

    def test_future_branch_is_queued_if_its_spawning_turn_is_already_past(self):
        graph = parse_trace_graph(
            trace(
                [
                    row(0, [1]),
                    {
                        "type": "subagent",
                        "agent_id": "child",
                        "t": 10,
                        "requests": [row(10, [4])],
                    },
                    row(20, [1, 2]),
                ]
            )
        )
        snap = snapshot_at(graph, 0, 5000)
        child = next(s for s in snap.states if s.conversation_id == "trace::sa:child")
        self.assertEqual(child.next_turn_index, 0)
        self.assertEqual(child.dispatch_offset_s, 5)
        self.assertIsNone(child.primer_turn_index)

    def test_no_next_request_returns_no_snapshot(self):
        self.assertIsNone(snapshot_at(self.graph(), 0, 100000))

    def test_sequential_pool_wraps_and_recycle_draw_is_shared(self):
        first = self.graph()
        raw = trace([row(0, [9]), row(10, [9, 10])])
        raw["id"] = "other"
        second = parse_trace_graph(raw)
        pool = SequentialTracePool([first, second])
        snapshots = initial_snapshots(pool, 3)
        self.assertEqual(
            [s.graph.trace_id for s in snapshots], ["trace", "other", "trace"]
        )
        self.assertEqual(pool.next_graph().trace_id, "other")
        self.assertNotEqual(snapshots[0].t_star_ms, snapshots[2].t_star_ms)

    def test_snapshot_seed_is_stable_and_independent_of_external_rng(self):
        graph = self.graph()
        first = sample_snapshot(graph, 5)
        random.seed(932)
        for _ in range(100):
            random.random()
        second = sample_snapshot(graph, 5)
        self.assertEqual(first, second)
        self.assertNotEqual(first.t_star_ms, sample_snapshot(graph, 6).t_star_ms)

    def test_primer_spread_caps_old_history_and_keeps_waiting_parent(self):
        snap = snapshot_at(self.graph(), 0, 4000)
        schedule = primer_schedule([snap], system_idle_cap_s=1)
        self.assertEqual(
            {key for _, _, key in schedule},
            {TurnKey("trace", 0), TurnKey("trace::sa:child", 0)},
        )
        self.assertTrue(all(0 <= offset <= 1 for offset, _, _ in schedule))

    def test_cache_scope_is_shared_with_children_and_changes_for_new_plays(self):
        original = cache_scope("benchmark", 0, 1, "trace")
        self.assertEqual(
            original, cache_scope("benchmark", 0, 1, "trace::sa:child:fa:000")
        )
        for args in (
            ("benchmark", 1, 1, "trace"),
            ("benchmark", 0, 2, "trace"),
            ("benchmark", 0, 1, "other"),
            ("different", 0, 1, "trace"),
        ):
            self.assertNotEqual(original, cache_scope(*args))


if __name__ == "__main__":
    unittest.main()
