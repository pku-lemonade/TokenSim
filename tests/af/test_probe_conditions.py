"""Condition optimization retains nested values, failures and callback cleanup."""

from contextlib import nullcontext
import unittest

import simpy

from scripts import af_probe_conditions


class ConditionProbeTest(unittest.TestCase):
    def test_nested_conditions_failure_and_processed_children(self):
        outcomes = []
        for optimized in (False, True):
            with af_probe_conditions.installed() if optimized else nullcontext():
                env = simpy.Environment()
                a, b, c = [env.timeout(time, time) for time in (0, 1, 2)]
                nested = (a & b) | c
                value = env.run(until=nested)
                observed = list(value.values())
                env.run()
                observed.append(list(env.run(until=env.all_of((a, b, c))).values()))
                observed.append(list(env.run(until=env.all_of(())).values()))
                pending = env.event()
                failed = env.event()
                condition = env.any_of((pending, failed))
                failed.fail(ValueError("expected failure"))
                with self.assertRaisesRegex(ValueError, "expected failure"):
                    env.run(until=condition)
                self.assertEqual(pending.callbacks, [])
                with self.assertRaisesRegex(ValueError, "different environments"):
                    env.all_of((env.event(), simpy.Environment().event()))
                outcomes.append(observed)
        self.assertEqual(outcomes[0], outcomes[1])
