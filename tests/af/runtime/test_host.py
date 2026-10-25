"""Scoped host GC policy must preserve event-loop and caller behavior."""

import gc
from threading import Event, Thread
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.trace.host import run_environment


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
