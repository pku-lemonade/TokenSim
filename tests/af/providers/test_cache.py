from dataclasses import replace
import json
import unittest

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.errors import ConfigurationError
from scripts.deepseek_v3_runtime import configuration, simulate
from tests.af.support.hardware import af_test_hardware
from tests.af.support.queries import _compute_query, _moe
from tests.af.support.services import observe_services


class EstimateCacheTest(unittest.TestCase):
    def test_repeated_query_is_immutable_and_preserves_provenance(self):
        hardware = af_test_hardware(moe=_moe())
        provider = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="analytical_only"
        )
        query = _compute_query(context=(8, 8))
        first = provider.estimate(query)
        lookups = provider.stats_dict()["operator_query_count"]
        second = provider.estimate(
            replace(query, iteration_id=12, block_id=4, layer_id=10)
        )
        self.assertIs(first, second)
        self.assertEqual(provider.stats_dict()["operator_query_count"], lookups)
        self.assertEqual(provider.estimate_cache_hits, 1)
        with self.assertRaises(TypeError):
            second.details["components"][0]["key"]["m"] = 999
        self.assertIn("components", json.loads(json.dumps(second.details)))
        changed = provider.estimate(replace(query, context_lengths=(16, 16)))
        self.assertIsNot(changed, first)
        self.assertEqual(provider.estimate_cache_hits, 1)

    def test_cache_does_not_mask_table_only_missing_shape(self):
        hardware = af_test_hardware(moe=_moe())
        model = hardware.model("TestModel")
        query = _compute_query(context=(8, 8))
        analytical = OperatorTableAFComputeProvider(
            hardware, model, fallback="analytical_only"
        )
        analytical.estimate(query)
        strict = OperatorTableAFComputeProvider(hardware, model, fallback="table_only")
        with self.assertRaisesRegex(
            ConfigurationError, "table_only.*requires operator data"
        ):
            strict.estimate(query)
        self.assertFalse(strict._estimate_cache)

    def test_cache_changes_lookup_only_not_scheduling_or_provenance(self):
        results, hits = [], []
        for enabled in (False, True):
            config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
            config.cluster.af.service_estimate_cache = enabled
            with observe_services() as services:
                _, rows, compute, transfer = simulate(
                    config, 2, 8, steps=3, fallback="analytical_only"
                )
            results.append((rows, services))
            hits.append((compute.estimate_cache_hits, transfer.estimate_cache_hits))
        self.assertEqual(results[0], results[1])
        self.assertEqual(hits[0], (0, 0))
        self.assertTrue(all(value > 0 for value in hits[1]))
