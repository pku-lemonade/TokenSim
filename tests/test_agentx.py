"""Trace validation, dependency clocks, recycling and cancellation semantics."""

import unittest

import simpy

from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.agentx import AgentXReplay, engine_hashes
from TokenSim.workload.agentx_support.parser import parse_trace_graph
from tests.support.replay import ControlledEngine


def row(time, hashes=(1,), output=2):
    return {
        "t": time,
        "type": "s",
        "model": "source-model",
        "in": 64 * len(hashes),
        "out": output,
        "hash_ids": list(hashes),
        "api_time": 0.1,
    }


def graph(rows, name="trace"):
    return parse_trace_graph(
        {"id": name, "block_size": 64, "hash_id_scope": "local", "requests": rows}
    )


class AgentXTest(unittest.TestCase):
    def test_hash_validation_and_block_conversion_preserve_prefix_proof(self):
        for invalid in (-1, True, 2**64, 1.5):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(WorkloadValidationError),
            ):
                graph([row(0, (invalid,))])
        hashes = engine_hashes(
            [1, 2, 3, 4, 5],
            trace_block_size=64,
            engine_block_size=256,
            input_tokens=319,
        )
        self.assertEqual(
            hashes,
            engine_hashes(
                [1, 2, 3, 4],
                trace_block_size=64,
                engine_block_size=256,
                input_tokens=256,
            ),
        )
        with self.assertRaises(WorkloadValidationError):
            engine_hashes(
                [1], trace_block_size=64, engine_block_size=32, input_tokens=64
            )

    def test_recorded_recycling_uses_each_trees_source_clock(self):
        env = simpy.Environment()
        engine = ControlledEngine(env)
        replay = AgentXReplay(
            env,
            engine,
            [graph([row(100), row(101)])],
            block_size=64,
            concurrency=1,
            profile_duration=3,
            warmup=False,
            issue_mode="recorded",
        )
        env.run(until=env.process(replay.run()))
        arrivals = [r.arrival_at for r in engine.submitted]
        self.assertGreaterEqual(len(arrivals), 5)
        for actual, expected in zip(arrivals[:4], [0, 1, 1.1, 2.1]):
            self.assertAlmostEqual(actual, expected)
        self.assertGreater(replay.play_count, 1)

    def test_deadline_cancels_inflight_without_generating_fake_output(self):
        env = simpy.Environment()
        engine = ControlledEngine(env, latency=10, cancel_delay=0.2)
        replay = AgentXReplay(
            env,
            engine,
            [graph([row(0, output=5)])],
            block_size=64,
            concurrency=1,
            profile_duration=1,
            grace_period=0,
            cancellation_timeout=1,
            warmup=False,
        )
        env.run(until=env.process(replay.run()))
        self.assertEqual(len(engine.submitted), 1)
        request = engine.submitted[0]
        self.assertTrue(request.cancel_requested)
        self.assertEqual(request.generation_idx, 0)
        self.assertIsNone(request.completed_at)
        self.assertEqual(replay.requests, [])
        self.assertLessEqual(env.now, 2)
        self.assertEqual(replay.cancel_unreturned, 0)
