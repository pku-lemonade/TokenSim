"""Scoped host GC policy must preserve event-loop and caller behavior."""

import gc
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.runtime.host import run_environment


class HostRunTest(unittest.TestCase):
    def setUp(self):
        self.previous = gc.get_threshold()
        self.enabled = gc.isenabled()
        gc.enable()
        gc.set_threshold(700, 10, 10)
        self.addCleanup(gc.set_threshold, *self.previous)
        self.addCleanup(gc.enable if self.enabled else gc.disable)

    def test_stop_resume_return_value_and_fifo_order(self):
        env = simpy.Environment()
        observations = []

        def task(value):
            observations.append((value, gc.get_threshold()))
            yield env.timeout(1)
            return value

        processes = [env.process(task(value)) for value in ("a", "b")]
        self.assertEqual(run_environment(env, until=processes[0]), "a")
        self.assertEqual([value for value, _ in observations], ["a", "b"])
        self.assertTrue(all(policy == (250_000, 10, 10) for _, policy in observations))
        self.assertEqual(gc.get_threshold(), (700, 10, 10))
        run_environment(env)
        self.assertTrue(all(process.triggered for process in processes))
        self.assertEqual(gc.get_threshold(), (700, 10, 10))

    def test_exception_restores_policy_and_custom_run_is_called(self):
        marker = object()
        self.assertIs(run_environment(SimpleNamespace(run=lambda: marker)), marker)

        def custom_run(*, until):
            self.assertIs(until, marker)
            self.assertEqual(gc.get_threshold(), (250_000, 10, 10))
            raise ValueError("custom failure")

        with self.assertRaisesRegex(ValueError, "custom failure"):
            run_environment(SimpleNamespace(run=custom_run), until=marker)
        self.assertEqual(gc.get_threshold(), (700, 10, 10))

    def test_disabled_collection_and_custom_thresholds(self):
        for enabled, policy in (
            (False, (700, 10, 10)),
            (True, (0, 10, 10)),
            (True, (1_000_000, 7, 9)),
        ):
            with self.subTest(enabled=enabled, policy=policy):
                (gc.enable if enabled else gc.disable)()
                gc.set_threshold(*policy)
                seen = []
                run_environment(
                    SimpleNamespace(
                        run=lambda: seen.append((gc.isenabled(), gc.get_threshold()))
                    )
                )
                self.assertEqual(seen, [(enabled, policy)])
                self.assertEqual(gc.get_threshold(), policy)

    def test_nested_calls_and_explicit_policy_changes(self):
        def outer():
            run_environment(simpy.Environment())
            self.assertEqual(gc.get_threshold(), (250_000, 10, 10))
            gc.set_threshold(1234, 8, 9)

        run_environment(SimpleNamespace(run=outer))
        self.assertEqual(gc.get_threshold(), (1234, 8, 9))

    def test_overlapping_runs_restore_only_after_both_exit(self):
        entered, release = Event(), Event()
        errors = []

        def waiting():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test run did not release")

        def worker():
            try:
                run_environment(SimpleNamespace(run=waiting))
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            run_environment(simpy.Environment())
            self.assertEqual(gc.get_threshold(), (250_000, 10, 10))
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(gc.get_threshold(), (700, 10, 10))

    def test_fixed_batch_scope_covers_submissions_between_environment_runs(self):
        from TokenSim.af.execution.model import ModelAFExecutor
        from scripts.deepseek_v3_runtime import configuration, simulate

        original = ModelAFExecutor.submit
        policies = []

        def submit(executor, *args, **kwargs):
            policies.append(gc.get_threshold())
            return original(executor, *args, **kwargs)

        with patch.object(ModelAFExecutor, "submit", submit):
            executor, _, _, _ = simulate(
                configuration(a_tp=1, f_tp=1, af_batch_size=1),
                1,
                8,
                steps=2,
                fallback="analytical_only",
            )
        executor.trace.close()
        self.assertEqual(policies, [(250_000, 10, 10)] * 2)
        self.assertEqual(gc.get_threshold(), (700, 10, 10))
