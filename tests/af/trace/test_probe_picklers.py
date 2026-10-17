"""Alternative picklers preserve sharing and nested immutable provider records."""

from dataclasses import replace
import pickle
import unittest

from TokenSim.af.caching.estimates import FrozenDict, freeze
from TokenSim.af.trace import archive_store
from scripts import af_probe_pickledispatch, af_probe_picklereduce
from tests.af.support.observations import _valid_events


class DerivedFrozenDict(FrozenDict):
    pass


class ProbePicklersTest(unittest.TestCase):
    def test_serialized_values_sharing_and_immutability_match_reference(self):
        details = freeze({"nested": [{"shape": [1, 2]}]})
        events = tuple(replace(event, details=details) for event in _valid_events())
        original = archive_store._encode(events)
        pickler = archive_store._TracePickler
        for probe in (af_probe_pickledispatch, af_probe_picklereduce):
            with self.subTest(probe=probe.__name__), probe.installed():
                encoded = archive_store._encode(events)
                self.assertEqual(encoded, original)
                restored = pickle.loads(encoded)
                self.assertEqual(restored, events)
                self.assertIs(restored[0].details, restored[1].details)
                with self.assertRaises(TypeError):
                    restored[0].details["nested"][0]["shape"] = ()
            self.assertIs(archive_store._TracePickler, pickler)

    def test_reduce_method_preserves_existing_subclass_canonicalization(self):
        value = DerivedFrozenDict({"nested": freeze({"x": [1]})})
        expected = archive_store._encode((value,))
        with af_probe_picklereduce.installed():
            self.assertEqual(archive_store._encode((value,)), expected)
            self.assertIs(type(pickle.loads(expected)[0]), FrozenDict)
