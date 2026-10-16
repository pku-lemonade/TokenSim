"""Optional Rust prototype contracts; build explicitly to enable these tests."""

from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

import simpy

from TokenSim.af.caching import estimates
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.trace.export import write_chrome_trace
from scripts.af_rust import backend
from tests.af.support.profiles import NativeFixture
from tests.support.hardware import make_model


@dataclass(frozen=True)
class NestedRecord:
    value: object


@unittest.skipUnless(backend.BINARY.exists(), "optional Rust probe has not been built")
class RustProbeTest(NativeFixture, unittest.TestCase):
    def compare(self, scenario):
        expected = scenario()
        with backend.installed(("keys", "templates", "events")):
            observed = scenario()
        self.assertEqual(expected, observed)

    def test_dense_and_moe_keep_rows_and_complete_trace_values(self):
        for model in (make_model(num_layers=2), self.model):
            results = []
            for native in (False, True):
                with self.subTest(model=model.model_id, rust=native):
                    context = backend.installed() if native else nullcontext()
                    with context, TemporaryDirectory() as directory:
                        runtime = build_model_runtime(
                            deepcopy(self.base.doc),
                            hardware=self.base.hardware,
                            model=model,
                        )
                        executor, rows, _, _ = simulate_decode(
                            runtime, 2, 16, steps=5, fallback="analytical_only"
                        )
                        try:
                            path = Path(directory) / "trace.json"
                            write_chrome_trace(path, executor.trace.events)
                            results.append((rows, json.loads(path.read_text())))
                        finally:
                            executor.trace.close()
            self.assertEqual(results[0], results[1])

    def test_native_keys_revisit_mutable_descendants_and_bound_storage(self):
        native = backend.library()
        memo = native.Fingerprints(
            backend._layout, Mapping, estimates.FrozenDict, capacity=4
        )
        mutable = {"values": [1, True, 1.0]}
        record = NestedRecord(mutable)
        before = memo(record)
        mutable["values"].append(2)
        self.assertNotEqual(before, memo(record))
        self.assertNotIn(id(record), memo.values)
        for value in range(40):
            nested = NestedRecord(((True, value, 1.0),))
            self.assertEqual(memo(nested), estimates.identity(nested))
            self.assertLessEqual(len(memo.values), 4)
        self.assertEqual(len({memo((value,)) for value in (True, 1, 1.0)}), 3)

    def test_same_time_dependencies_resource_fifo_and_interrupts(self):
        def scenario():
            env = simpy.Environment()
            resource = simpy.Resource(env, 1)
            rows = []

            def worker(index):
                try:
                    with resource.request() as request:
                        yield request
                        rows.append((index, "start", env.now))
                        yield env.timeout(1)
                        rows.append((index, "end", env.now))
                except simpy.Interrupt as exc:
                    rows.append((index, exc.cause, env.now))

            workers = [env.process(worker(index)) for index in range(4)]

            def cancel():
                yield env.timeout(0.5)
                workers[2].interrupt("cancelled")

            env.process(cancel())
            env.run(until=env.all_of(workers))
            env.run()
            return rows, env.now

        self.compare(scenario)

    def test_numeric_deadlines_and_processed_event_resume(self):
        def scenario():
            env = simpy.Environment(initial_time=0.1)
            env.run(until=1)
            event = env.timeout(0.5, "value")
            rows = []
            event.callbacks.append(lambda event: rows.append("before"))
            result = env.run(until=event)
            event.callbacks.append(lambda event: rows.append("after"))
            env.run()
            return result, env.run(until=event), rows, env.now

        self.compare(scenario)

    def test_callback_list_growth_keeps_python_iteration_order(self):
        def scenario():
            env = simpy.Environment()
            event = env.event()
            callbacks = event.callbacks
            rows = []

            def append(_event):
                rows.append("append")
                callbacks.append(lambda _event: rows.append("added"))

            callbacks.append(append)
            event.succeed()
            env.run()
            return rows

        self.compare(scenario)

    def test_failure_copies_and_exhausted_until_event(self):
        for native in (False, True):
            context = backend.installed(("events",)) if native else nullcontext()
            with context:
                env = simpy.Environment()
                error = ValueError("failed")
                env.event().fail(error)
                with self.assertRaisesRegex(ValueError, "failed") as raised:
                    env.run()
                self.assertIs(raised.exception.__cause__, error)
                with self.assertRaisesRegex(RuntimeError, "No scheduled events"):
                    env.run(until=env.event())
                with self.assertRaisesRegex(ValueError, "greater than"):
                    env.run(until=env.now)

    def test_event_temporaries_are_released_during_long_run(self):
        references = []
        retained = []
        with backend.installed(("events",)):
            env = simpy.Environment()

            def worker():
                for index in range(2000):
                    event = env.timeout(0)
                    references.append(weakref.ref(event))
                    yield event
                    if index % 250 == 0:
                        retained.append(sum(ref() is not None for ref in references))

            env.process(worker())
            env.run()
        gc.collect()
        self.assertLess(max(retained), 20)
        self.assertFalse(any(ref() is not None for ref in references))

    def test_installation_restores_functions_and_keeps_custom_step(self):
        original_run = simpy.Environment.run
        original_memo = estimates.FingerprintMemo
        with self.assertRaisesRegex(ValueError, "probe failure"):
            with backend.installed(("events",)):
                env = simpy.Environment()
                count = []
                original_step = env.step

                def step():
                    count.append(env.now)
                    return original_step()

                env.step = step
                env.timeout(1)
                env.run()
                self.assertTrue(count)
                raise ValueError("probe failure")
        self.assertIs(simpy.Environment.run, original_run)
        self.assertIs(estimates.FingerprintMemo, original_memo)

        original_step = simpy.Environment.step
        calls = []

        def customized_step(self):
            calls.append(self.now)
            return original_step(self)

        with patch.object(simpy.Environment, "step", customized_step):
            with backend.installed(("events",)):
                env = simpy.Environment()
                env.timeout(1)
                env.run()
        self.assertTrue(calls)
