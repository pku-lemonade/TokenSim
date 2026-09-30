from __future__ import annotations

import unittest

import simpy

from TokenSim.config.config import ClusterConfig, ParallelConfig, WorkerGroupConfig
from TokenSim.errors import SimulationStateError
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, RequestStatus, reset_g_time
from TokenSim.workload.agentx_native.parser import parse_trace_graph
from TokenSim.workload.agentx_native.replay import NativeAgentXReplay, engine_hashes
from tests.test_parallel_simulation import _hardware, _psla


def graph(rows, tid="trace"):
    return parse_trace_graph(
        {"id": tid, "block_size": 64, "hash_id_scope": "local", "requests": rows}
    )


def row(t, hashes, *, api=0.1, out=2):
    return {
        "type": "s",
        "t": t,
        "in": len(hashes) * 64,
        "out": out,
        "hash_ids": hashes,
        "model": "model",
        "api_time": api,
    }


class ControlledEngine:
    def __init__(self, env, latency=0.1, *, cancel_delay=0, error_ids=()):
        self.env, self.latency = env, latency
        self.cancel_delay = cancel_delay
        self.error_ids = set(error_ids)
        self.submitted = []
        self.reset_count = 0

    def add_requests(self, requests):
        for req in requests:
            self.submitted.append(req)
            self.env.process(self._run(req))

    def _run(self, req):
        yield self.env.timeout(
            self.latency(req) if callable(self.latency) else self.latency
        )
        if req.cancel_requested:
            return
        if req.id in self.error_ids:
            req.completion_event.fail(RuntimeError("controlled request error"))
            return
        for _ in range(req.decode_len):
            req.step(self.env, 0, 1)

    def cancel_requests(self, requests):
        events = []
        for req in requests:
            events.append(req.request_cancel(self.env))
            self.env.process(self._cancel(req))
        return events

    def _cancel(self, req):
        yield self.env.timeout(self.cancel_delay)
        req.acknowledge_cancel(self.env)

    def reset_profile_stats(self):
        self.reset_count += 1


def real_engine(env, *, latency=None):
    parallel = ParallelConfig()
    cluster = ClusterConfig(
        num_workers=1,
        networks={"node0": "ethernet-test"},
        worker_groups=[
            WorkerGroupConfig(
                role="hybrid", hardware="TestGPU", num_workers=1, network="node0"
            )
        ],
    )
    engine = LLMEngine(
        env=env,
        block_size=256,
        batching="paged-attn",
        kv_transfer_config=cluster.effective_kv_transfer(),
        psla_config=_psla(),
        cluster_config=cluster,
        parallel_config=parallel,
        hardware=_hardware(),
        prefill_worker_pool_type="round_robin",
        decode_worker_pool_type="round_robin",
        max_parallem_sum=8,
        latency_backend_type="analytical",
    )
    if latency is not None:
        engine.workers[0].dynamic_batch = lambda requests: latency
    return engine


class CancellationLifecycleTest(unittest.TestCase):
    def setUp(self):
        reset_g_time()

    def test_cancelled_request_does_not_fake_output_or_success(self):
        env = simpy.Environment()
        req = Request(1, 256, 10, 256)
        req.arrive(env)
        ack = req.request_cancel(env)
        req.step(env, 1, 1)
        req.advance(env, 1, 1)
        self.assertEqual(req.generation_idx, 0)
        self.assertFalse(req.is_done)
        self.assertFalse(ack.triggered)
        req.acknowledge_cancel(env)
        self.assertTrue(ack.triggered)
        self.assertEqual(req.status, RequestStatus.FINISHED_CANCELLED)
        self.assertIsNone(req.completed_at)

    def test_cancel_before_engine_dispatch_never_allocates_or_generates(self):
        env = simpy.Environment()
        engine = real_engine(env)
        req = Request(0, 256, 10, 256)
        req.arrive(env)
        engine.add_requests([req])
        ack = engine.cancel_requests([req])[0]
        env.run(until=0.01)
        self.assertTrue(ack.triggered)
        self.assertEqual(req.generation_idx, 0)
        self.assertEqual(
            engine.workers[0].scheduler.block_manager.block_table.get_blocks(0), []
        )
        engine.raise_if_failed()

    def test_cancel_queued_on_worker_is_idempotent(self):
        env = simpy.Environment()
        engine = real_engine(env)
        req = Request(0, 256, 10, 256)
        req.arrive(env)
        engine.add_requests_burst([req])
        first = engine.cancel_requests([req])[0]
        second = engine.cancel_requests([req])[0]
        self.assertIs(first, second)
        env.run(until=0.01)
        self.assertEqual(req.status, RequestStatus.FINISHED_CANCELLED)
        self.assertEqual(engine.workers[0].scheduler.running, [])
        self.assertEqual(engine.workers[0].scheduler.waiting, [])
        engine.raise_if_failed()

    def test_kernel_holds_blocks_until_boundary_and_cancel_prevents_late_sampling(self):
        env = simpy.Environment()
        engine = real_engine(env, latency=5)
        req = Request(0, 256, 10, 256)
        req.arrive(env)
        engine.add_requests_burst([req])
        env.run(until=1)
        manager = engine.workers[0].scheduler.block_manager
        self.assertTrue(manager.block_table.get_blocks(0))
        ack = engine.cancel_requests([req])[0]
        env.run(until=4)
        self.assertFalse(ack.triggered)
        self.assertTrue(manager.block_table.get_blocks(0))
        env.run(until=6)
        self.assertTrue(ack.triggered)
        self.assertEqual(req.generation_idx, 0)
        self.assertEqual(manager.block_table.get_blocks(0), [])
        self.assertEqual(manager.gpu_allocator.get_num_allocated_blocks(), 0)
        engine.raise_if_failed()

    def test_successful_request_remains_successful_when_cancel_arrives_late(self):
        env = simpy.Environment()
        req = Request(1, 64, 1, 256)
        req.arrive(env)
        req.step(env, 0, 1)
        event = req.request_cancel(env)
        self.assertTrue(event.triggered)
        self.assertTrue(req.is_done)
        self.assertFalse(req.cancel_requested)


class NativeReplayLifecycleTest(unittest.TestCase):
    def setUp(self):
        reset_g_time()

    def run_replay(self, graphs, **options):
        env = simpy.Environment()
        engine = ControlledEngine(
            env,
            options.pop("latency", 0.1),
            cancel_delay=options.pop("cancel_delay", 0),
            error_ids=options.pop("error_ids", ()),
        )
        replay = NativeAgentXReplay(
            env,
            engine,
            graphs,
            concurrency=options.pop("concurrency", 1),
            profile_duration=options.pop("profile_duration", 2),
            **options,
        )
        process = env.process(replay.run())
        env.run(until=process)
        return env, engine, replay

    def test_hash_aggregation_keeps_complete_prefix_chains_and_drops_partial_tail(self):
        first = engine_hashes(
            [1, 2, 3, 4, 5, 6, 7, 8],
            trace_block_size=64,
            engine_block_size=256,
            input_tokens=512,
        )
        extended = engine_hashes(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            trace_block_size=64,
            engine_block_size=256,
            input_tokens=600,
        )
        self.assertEqual(first, extended)
        changed = engine_hashes(
            [1, 2, 3, 4, 5, 6, 7, 99],
            trace_block_size=64,
            engine_block_size=256,
            input_tokens=512,
        )
        self.assertEqual(first[0], changed[0])
        self.assertNotEqual(first[1], changed[1])

    def test_warmup_has_primers_plus_exact_per_lane_quota_and_handoff_does_not_repeat(
        self,
    ):
        trees = [
            graph([row(i, [1] * (i + 1)) for i in range(30)], "one"),
            graph([row(i, [2] * (i + 1)) for i in range(30)], "two"),
        ]
        env, engine, replay = self.run_replay(
            trees, concurrency=2, warmup=True, profile_duration=2
        )
        self.assertEqual(replay.pressure_sent, [10, 10])
        self.assertEqual(len(replay.warmup_requests), 22)
        self.assertEqual(engine.reset_count, 1)
        self.assertTrue(all(r.decode_len == 1 for r in replay.warmup_requests))
        warm = {
            (
                r.agentx_trace_id,
                r.agentx_lane,
                r.agentx_play_id,
                r.agentx_conversation_id,
                r.turn,
            )
            for r in replay.warmup_requests
        }
        profile = {
            (
                r.agentx_trace_id,
                r.agentx_lane,
                r.agentx_play_id,
                r.agentx_conversation_id,
                r.turn,
            )
            for r in replay.requests
        }
        self.assertFalse(warm & profile)
        for req in replay.requests:
            same_tree = [
                r
                for r in replay.warmup_requests
                if r.agentx_trace_id == req.agentx_trace_id
                and r.agentx_play_id == req.agentx_play_id
            ]
            if same_tree:
                self.assertEqual(req.cache_salt, same_tree[0].cache_salt)

    def test_closed_loop_wait_excludes_historical_service_time(self):
        tree = graph([row(0, [1], api=2), row(5, [1, 2], api=1)])
        env, engine, replay = self.run_replay(
            [tree], warmup=False, latency=1, profile_duration=4.5
        )
        same_tree = [r for r in engine.submitted if r.agentx_play_id == 0]
        self.assertEqual([r.arrival_at for r in same_tree], [0, 4])

    def test_grace_completion_is_included_and_no_request_is_issued_after_window(self):
        env, engine, replay = self.run_replay(
            [graph([row(0, [1])])],
            warmup=False,
            latency=2,
            profile_duration=1,
            grace_period=3,
        )
        self.assertEqual(len(engine.submitted), 1)
        self.assertEqual(len(replay.requests), 1)
        self.assertEqual(replay.requests[0].completed_at, 2)
        self.assertEqual(replay.profile_elapsed, 2)

    def test_grace_then_cancel_waits_for_ack_and_retains_cancelled_accounting(self):
        env, engine, replay = self.run_replay(
            [graph([row(0, [1])])],
            warmup=False,
            latency=100,
            profile_duration=1,
            grace_period=3,
            cancellation_timeout=2,
            cancel_delay=1,
        )
        self.assertEqual(replay.profile_elapsed, 5)
        self.assertEqual(replay.phase_counts["profiling"]["sent"], 1)
        self.assertEqual(replay.phase_counts["profiling"]["cancelled"], 1)
        self.assertEqual(replay.requests, [])
        self.assertEqual(replay.records[0]["outcome"], "cancelled")

    def test_cancel_timeout_is_bounded_and_late_callbacks_do_not_double_count(self):
        env, engine, replay = self.run_replay(
            [graph([row(0, [1])])],
            warmup=False,
            latency=100,
            profile_duration=1,
            grace_period=3,
            cancellation_timeout=2,
            cancel_delay=50,
        )
        self.assertEqual(replay.profile_elapsed, 6)
        self.assertEqual(replay.cancel_unreturned, 1)
        self.assertEqual(len(replay.records), 1)
        self.assertFalse(replay.records[0]["cancellation_acknowledged"])
        env.run(until=105)
        self.assertEqual(len(engine.submitted), 1)
        self.assertEqual(replay.phase_counts["profiling"]["cancelled"], 1)
        self.assertEqual(replay.phase_counts["profiling"]["successful"], 0)

    def test_errors_advance_dependencies_but_are_excluded_from_success_metrics(self):
        env, engine, replay = self.run_replay(
            [graph([row(0, [1]), row(1, [1, 2])])],
            warmup=False,
            error_ids=[0],
            profile_duration=1.1,
        )
        self.assertEqual(replay.phase_counts["profiling"]["error"], 1)
        self.assertTrue(any(r.turn == 1 for r in engine.submitted))
        self.assertTrue(all(r.id != 0 for r in replay.requests))

    def test_tree_idle_cap_advances_its_timers_while_other_tree_remains_busy(self):
        trees = [
            graph([row(0, [1]), row(1000, [1, 2])], "idle"),
            graph([row(0, [2])], "busy"),
        ]
        env, engine, replay = self.run_replay(
            trees,
            concurrency=2,
            warmup=False,
            profile_duration=305,
            tree_idle_cap=300,
            system_idle_cap=10,
            latency=lambda r: 1000 if r.agentx_trace_id == "busy" else 0.1,
        )
        idle_turns = [
            r
            for r in engine.submitted
            if r.agentx_trace_id == "idle" and r.agentx_play_id == 0
        ]
        self.assertEqual([r.turn for r in idle_turns], [0, 1])
        self.assertAlmostEqual(idle_turns[1].arrival_at, 300.1)
        self.assertGreaterEqual(replay.tree_idle_jumps, 1)

    def test_system_idle_shift_preserves_spacing_between_pending_trees(self):
        trees = [
            graph([row(0, [1]), row(100, [1, 2])], "a"),
            graph([row(0, [2]), row(110, [2, 3])], "b"),
        ]
        env, engine, replay = self.run_replay(
            trees,
            concurrency=2,
            warmup=False,
            profile_duration=25,
            system_idle_cap=10,
            tree_idle_cap=300,
        )
        first = {
            r.agentx_trace_id: r
            for r in engine.submitted
            if r.agentx_play_id == 0 and r.turn == 1
        }
        self.assertAlmostEqual(first["a"].arrival_at, 10.1)
        self.assertAlmostEqual(first["b"].arrival_at, 20.1)
        self.assertGreater(replay.system_idle_jumps, 0)

    def test_real_engine_replay_cancels_inflight_work_without_freeing_early(self):
        env = simpy.Environment()
        engine = real_engine(env, latency=5)
        replay = NativeAgentXReplay(
            env,
            engine,
            [graph([row(0, [1])])],
            concurrency=1,
            warmup=False,
            profile_duration=1,
            grace_period=1,
            cancellation_timeout=5,
        )
        process = env.process(replay.run())
        env.run(until=process)
        self.assertEqual(replay.phase_counts["profiling"]["cancelled"], 1)
        self.assertEqual(env.now, 5)
        self.assertEqual(
            engine.workers[
                0
            ].scheduler.block_manager.gpu_allocator.get_num_allocated_blocks(),
            0,
        )
        engine.raise_if_failed()

    def test_real_engine_success_and_logging_sinks_do_not_change_replay(self):
        results = []
        tree = graph([row(0, [1, 2, 3, 4]), row(1, [1, 2, 3, 4, 5])])
        for enabled in (False, True):
            env = simpy.Environment()
            engine = real_engine(env)
            events, records = [], []
            replay = NativeAgentXReplay(
                env,
                engine,
                [tree],
                concurrency=1,
                warmup=False,
                profile_duration=0.1,
                event_sink=events.append if enabled else None,
                request_sink=records.append if enabled else None,
            )
            process = env.process(replay.run())
            env.run(until=process)
            engine.raise_if_failed()
            results.append(
                (
                    replay.phase_counts,
                    [(r.id, r.arrival_at, r.completed_at) for r in replay.requests],
                )
            )
            self.assertTrue(replay.requests)
            self.assertTrue(
                all(
                    not r._logical_token_blocks and r.hash_ids is None
                    for r in replay.requests
                )
            )
            if enabled:
                self.assertTrue(events and records)
        self.assertEqual(results[0], results[1])

    def test_native_empty_metric_set_is_missing_not_zero(self):
        from argparse import Namespace
        from util.results import get_agentx_metrics
        from TokenSim.config.psla_config import MetricData

        args = Namespace(
            workload_type="agentx_weka",
            agentx_concurrency=1,
            agentx_runtime_metadata={"replay_protocol": "aiperf-49634d1c-native"},
        )
        metrics = get_agentx_metrics(args, [], 3600, num_gpus=8)
        self.assertIsNone(metrics["ttft_s"]["p90"])
        self.assertFalse(metrics["metric_observation_defined"])
        self.assertEqual(
            metrics["throughput_denominator_kind"], "phase-elapsed-no-success"
        )
        self.assertIsNone(MetricData.from_list([]).max)


if __name__ == "__main__":
    unittest.main()
