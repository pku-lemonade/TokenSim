"""Differential interval tests retain general capacities and error diagnostics."""

import random
from types import SimpleNamespace
import unittest

from TokenSim.af.trace.occupancy import validate_resource_capacity as reference
from scripts.af_probe_validation import validate_resource_capacity as candidate
from TokenSim.errors import SimulationStateError


class IntervalValidationTest(unittest.TestCase):
    def test_shuffled_intervals_and_capacities_match_reference(self):
        random_source = random.Random(912)
        for case in range(1200):
            events = []
            for index in range(random_source.randrange(20)):
                start = random_source.randrange(8)
                end = start + random_source.randrange(5)
                events.append(
                    SimpleNamespace(
                        start_at=start,
                        end_at=end,
                        resource_end_at=None,
                        details={},
                        resource_ids=("r",),
                        resource_id="r",
                        node_id=str(index),
                        iteration_id=case,
                    )
                )
            random_source.shuffle(events)
            capacities = {"r": random_source.randrange(5)}
            outcomes = []
            for function in (reference, candidate):
                try:
                    function(events, capacities)
                    outcomes.append(None)
                except SimulationStateError as exc:
                    outcomes.append(str(exc))
            self.assertEqual(outcomes[0], outcomes[1], (events, capacities))
