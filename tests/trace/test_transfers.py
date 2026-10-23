"""Native queued transfers preserve service, queueing and partial visibility."""

from types import SimpleNamespace
import unittest

import simpy

from TokenSim.kv_transfer.native_storage import QueuedKVLink
from TokenSim.trace.recorder import TraceRecorder
from TokenSim.trace.transfers import TransferObserver


class QueuedTransferTraceTest(unittest.TestCase):
    def simulation(self, tracing):
        env = simpy.Environment()
        link = QueuedKVLink(env, 10)
        recorder = TraceRecorder()
        session = SimpleNamespace(
            engine=SimpleNamespace(env=env),
            recorder=recorder,
            iteration=0,
            run_phase="profiling",
            _complete=recorder.validate_invariants,
        )
        connector = SimpleNamespace(
            _metadata=SimpleNamespace(loads=[]), start_load_kv=lambda: 0, read_link=link
        )
        observer = TransferObserver(session)
        if tracing:
            observer.attach(SimpleNamespace(id=0, connector=connector))
        for _ in range(2):
            env.process(link.transfer(10))
        return env, link, recorder, observer

    def test_observation_preserves_shared_queue_timing(self):
        outcomes = []
        for tracing in (False, True):
            env, link, recorder, _ = self.simulation(tracing)
            env.run()
            outcomes.append((env.now, link.describe()))
            if tracing:
                self.assertEqual(len(recorder.events), 2)
                self.assertEqual([event.start_at for event in recorder.events], [0, 1])
                self.assertEqual(
                    [event.queue_duration for event in recorder.events], [0, 1]
                )
                self.assertEqual(
                    [event.payload_bytes for event in recorder.events], [10, 10]
                )
        self.assertEqual(outcomes[0], outcomes[1])

    def test_partial_dma_never_claims_full_payload_completion(self):
        env, _, recorder, observer = self.simulation(True)
        env.run(until=0.25)
        self.assertFalse(recorder.events)
        partial = list(observer.partial_events())
        self.assertEqual(len(partial), 1)
        self.assertEqual(partial[0].end_at, 0.25)
        self.assertIsNone(partial[0].payload_bytes)
        self.assertTrue(partial[0].details["incomplete"])
