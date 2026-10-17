"""Run the existing event-loop contracts against the Python experiment."""

import unittest
from unittest.mock import patch

import simpy

from scripts import af_probe_loop
from scripts.af_rust import backend
from tests.af import test_rust_probe as contracts


@unittest.skipUnless(simpy.__version__ == "4.1.2", "probe requires SimPy 4.1.2")
class PythonLoopProbeTest(unittest.TestCase):
    def test_existing_event_contracts(self):
        names = (
            "test_same_time_dependencies_resource_fifo_and_interrupts",
            "test_numeric_deadlines_and_processed_event_resume",
            "test_callback_list_growth_keeps_python_iteration_order",
            "test_failure_copies_and_exhausted_until_event",
            "test_event_temporaries_are_released_during_long_run",
            "test_installation_restores_functions_and_keeps_custom_step",
        )
        for name in names:
            with self.subTest(contract=name):
                with patch.object(
                    backend, "installed", lambda *a: af_probe_loop.installed()
                ):
                    getattr(contracts.RustProbeTest(), name)()
