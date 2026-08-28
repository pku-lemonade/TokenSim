from __future__ import annotations

import unittest
from pathlib import Path

import simpy

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import reset_g_time
from TokenSim.placement.policies import DataParallelWorkerPool, RoundRobinWorkerPool
from TokenSim.workload.agentx import (
    _SystemIdleCoordinator,
    AgentXReplay,
    load_agentx_traces,
)


FIXTURE = Path(__file__).parent / "fixtures" / "agentx_weka.jsonl"


class _FakeEngine:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.reset_count = 0

    def add_requests(self, requests):
        for request in requests:
            self.env.process(self._complete(request))

    def _complete(self, request):
        for _ in range(request.decode_len):
            yield self.env.timeout(0.01)
            request.step(self.env, 0.01, 1)

    def reset_profile_stats(self):
        self.reset_count += 1


class _Worker:
    def __init__(self, worker_id: int, dp_rank: int):
        self.id = worker_id
        self.dp_rank = dp_rank


class AgentXLoaderTest(unittest.TestCase):
    def test_loads_jsonl_subagents_and_uint64_hashes(self):
        traces = load_agentx_traces(str(FIXTURE), trace_count=2)

        self.assertEqual([trace.trace_id for trace in traces], ["trace-a", "trace-b"])
        self.assertEqual(traces[0].request_count, 5)
        self.assertEqual(len(traces[0].subagents[0].streams), 2)
        self.assertEqual(
            traces[1].roots[0].hash_ids[0],
            18446744073709551615,
        )

    def test_loader_preserves_recorded_trace_timestamps(self):
        trace = load_agentx_traces(str(FIXTURE), trace_count=1)[0]

        self.assertEqual(trace.roots[1].t, 5.0)
        self.assertEqual(trace.subagents[0].end_t, 3.0)

    def test_rejects_block_size_mismatch(self):
        trace = load_agentx_traces(str(FIXTURE), trace_count=1)[0]
        env = simpy.Environment()
        with self.assertRaisesRegex(WorkloadValidationError, "block_size"):
            AgentXReplay(env, _FakeEngine(env), [trace], block_size=16, concurrency=1)


class AgentXReplayTest(unittest.TestCase):
    def setUp(self):
        reset_g_time()

    def test_closed_loop_replay_waits_from_previous_completion(self):
        trace = load_agentx_traces(str(FIXTURE), trace_count=1)[0]
        env = simpy.Environment()
        engine = _FakeEngine(env)
        replay = AgentXReplay(
            env,
            engine,
            [trace],
            block_size=64,
            concurrency=1,
            warmup=False,
        )

        env.process(replay.run())
        env.run()

        roots = [
            request for request in replay.requests if request.agentx_stream_id == "root"
        ]
        self.assertEqual(len(replay.requests), 5)
        self.assertAlmostEqual(roots[0].arrival_at, 0.0)
        self.assertAlmostEqual(roots[1].arrival_at, 4.52)
        self.assertEqual(replay.play_count, 1)
        self.assertTrue(all(request.is_done for request in replay.requests))
        self.assertTrue(
            all(request.cache_salt == "agentx-play-0-0" for request in replay.requests)
        )

    def test_warmup_is_excluded_and_resets_profile_counters(self):
        trace = load_agentx_traces(str(FIXTURE), trace_count=1)[0]
        env = simpy.Environment()
        engine = _FakeEngine(env)
        replay = AgentXReplay(
            env,
            engine,
            [trace],
            block_size=64,
            concurrency=1,
            warmup=True,
            warmup_min_ratio=0.0,
            warmup_max_ratio=0.0,
        )

        env.process(replay.run())
        env.run()

        self.assertEqual(engine.reset_count, 1)
        self.assertEqual(len(replay.warmup_requests), 1)
        self.assertTrue(all(request.record_timing for request in replay.requests))
        self.assertTrue(
            all(not request.record_timing for request in replay.warmup_requests)
        )

    def test_empty_subagent_keeps_lane_open_until_join_barrier(self):
        trace = load_agentx_traces(str(FIXTURE), trace_count=1, trace_skip_count=1)[0]
        env = simpy.Environment()
        replay = AgentXReplay(
            env,
            _FakeEngine(env),
            [trace],
            block_size=64,
            concurrency=1,
            warmup=False,
        )

        env.process(replay.run())
        env.run()

        self.assertEqual(len(replay.requests), 1)
        self.assertAlmostEqual(replay.profile_elapsed, 1.11)

    def test_profile_deadline_excludes_late_completions(self):
        trace = load_agentx_traces(str(FIXTURE), trace_count=1)[0]
        env = simpy.Environment()
        replay = AgentXReplay(
            env,
            _FakeEngine(env),
            [trace],
            block_size=64,
            concurrency=1,
            profile_duration=0.005,
            warmup=False,
        )

        env.process(replay.run())
        env.run()

        self.assertAlmostEqual(replay.profile_elapsed, 0.005)
        self.assertEqual(replay.requests, [])

    def test_system_idle_cap_only_shifts_timers_when_all_requests_are_idle(self):
        env = simpy.Environment()
        coordinator = _SystemIdleCoordinator(env, 10.0)
        coordinator.request_started()
        delayed = coordinator.delay(100.0)
        observed = []

        def finish_request():
            yield env.timeout(20.0)
            coordinator.request_finished()

        def observe_delay():
            yield delayed
            observed.append(env.now)

        env.process(finish_request())
        env.process(observe_delay())
        env.run()

        self.assertEqual(observed, [30.0])

    def test_system_idle_cap_preserves_spacing_between_pending_timers(self):
        env = simpy.Environment()
        coordinator = _SystemIdleCoordinator(env, 10.0)
        first = coordinator.delay(100.0)
        second = coordinator.delay(110.0)
        observed = []

        def observe(label, event):
            yield event
            observed.append((label, env.now))
            if label == "first":
                coordinator.request_started()
                coordinator.request_finished()

        env.process(observe("first", first))
        env.process(observe("second", second))
        env.run()

        self.assertEqual(observed, [("first", 10.0), ("second", 20.0)])


class SessionAffinityTest(unittest.TestCase):
    def test_requests_from_one_agentx_lane_keep_the_same_dp_rank(self):
        workers = [_Worker(0, 0), _Worker(1, 1)]
        pool = DataParallelWorkerPool(workers, RoundRobinWorkerPool)
        first = type("Request", (), {"chat_id": "lane-7", "dp_rank": None})()
        second = type("Request", (), {"chat_id": "lane-7", "dp_rank": None})()

        first_worker = pool.select_prefill_worker(first)
        second_worker = pool.select_prefill_worker(second)

        self.assertEqual(first_worker.dp_rank, second_worker.dp_rank)
        self.assertEqual(first.dp_rank, second.dp_rank)


if __name__ == "__main__":
    unittest.main()
