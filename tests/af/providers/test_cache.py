"""Service cache identity, immutable configuration and uncached equivalence."""

from dataclasses import FrozenInstanceError, replace
from functools import wraps
import json
import unittest
from unittest.mock import patch

from TokenSim.af.caching.estimates import PreparedQuery, prepare_query
from TokenSim.af.caching.namespace import EstimateNamespace
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.af.providers.operator_table import (
    OperatorTableAFComputeProvider,
    TopologyAFTransferProvider,
)
from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.operator_data.lookup import LookupPolicy
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from scripts.run_deepseek_mla import configuration, simulate
from tests.af.support.config import _cluster_dict
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _compute_query, _moe, _transfer_query
from tests.af.support.observations import observe_services


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


class CacheFastPathTest(unittest.TestCase):
    def compute(self, **kwargs):
        hardware = af_test_hardware(moe=_moe())
        return OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), **kwargs
        )

    def transfer(self):
        cluster = ClusterConfig(**_cluster_dict(attention_workers=2, ffn_workers=2))
        hardware = af_test_hardware()
        workers = cluster.workers()
        placement = build_topology_placement(cluster, workers, hardware)
        return TopologyAFTransferProvider(hardware, cluster, workers, placement)

    def prepared(self, provider, query):
        return prepare_query(provider, type(query), **vars(query))

    def test_hits_skip_materialization_for_compute_and_transfer(self):
        for provider, query in (
            (self.compute(), _compute_query()),
            (self.transfer(), _transfer_query()),
        ):
            with self.subTest(provider=type(provider).__name__):
                first = provider.estimate(self.prepared(provider, query))
                self.assertEqual(provider.query_constructions, 1)
                with patch.object(
                    PreparedQuery,
                    "materialize",
                    side_effect=AssertionError("unexpected construction"),
                ):
                    self.assertIs(
                        first,
                        provider.estimate(
                            self.prepared(provider, replace(query, iteration_id=9))
                        ),
                    )
                    self.assertIs(first, provider.estimate(query))
                self.assertEqual(provider.estimate_cache_hits, 2)
                self.assertEqual(provider.query_constructions, 1)
                provider.cache_enabled = False
                full = self.prepared(provider, query)
                self.assertIs(type(full), type(query))
                self.assertEqual(provider.estimate(full), first)

    def test_invalid_types_cannot_hit_integer_entries(self):
        compute = self.compute()
        valid = _compute_query(tokens=1)
        compute.estimate(self.prepared(compute, valid))
        for invalid in (True, 1.0):
            query = replace(valid, local_shape={**valid.local_shape, "tokens": invalid})
            with self.assertRaisesRegex(ValueError, "positive integers"):
                compute.estimate(self.prepared(compute, query))
        transfer = self.transfer()
        transfer.estimate(self.prepared(transfer, _transfer_query(payload_bytes=1)))
        for invalid in (True, 1.0):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                transfer.estimate(
                    self.prepared(transfer, _transfer_query(payload_bytes=invalid))
                )
        self.assertEqual(compute.estimate_cache_hits, 0)
        self.assertEqual(transfer.estimate_cache_hits, 0)

    def test_mutable_query_inputs_are_not_memoized(self):
        provider = self.compute()
        query = _compute_query()
        provider.estimate(self.prepared(provider, query))
        query.local_shape["tokens"] = 99
        with self.assertRaisesRegex(ValueError, "does not match"):
            provider.estimate(self.prepared(provider, query))
        self.assertEqual(provider.estimate_cache_hits, 0)

    def test_compact_and_full_keys_agree_across_dynamic_fields(self):
        compute = self.compute(fallback="analytical_only")
        base = _compute_query()
        queries = [
            base,
            replace(base, context_lengths=(16,)),
            replace(base, dtype="bf16"),
            replace(base, physical_worker="a-pool:1"),
            replace(base, phase="prefill"),
            _compute_query(tokens=4),
            _compute_query("expert", expert_workload=((1, 2),)),
        ]
        for query in queries:
            before = compute.estimate_cache_hits
            full = compute.estimate(query)
            self.assertEqual(compute.estimate_cache_hits, before)
            self.assertIs(full, compute.estimate(self.prepared(compute, query)))
        transfer = self.transfer()
        base = _transfer_query()
        for query in (
            base,
            replace(base, payload_bytes=8192),
            replace(base, participant_ranks=(1, 0)),
            replace(base, physical_endpoints=("a-pool:1", "f-pool:1")),
            replace(base, source_layout="replicated"),
        ):
            before = transfer.estimate_cache_hits
            full = transfer.estimate(query)
            self.assertEqual(transfer.estimate_cache_hits, before)
            self.assertIs(full, transfer.estimate(self.prepared(transfer, query)))

    def test_subclass_and_wrapped_estimate_receive_full_queries(self):
        class CustomProvider(OperatorTableAFComputeProvider):
            def estimate(self, query):
                self.received = query
                return super().estimate(query)

        hardware = af_test_hardware(moe=_moe())
        provider = CustomProvider(hardware, hardware.model("TestModel"))
        query = self.prepared(provider, _compute_query())
        provider.estimate(query)
        self.assertIs(type(provider.received), AFComputeQuery)
        original = OperatorTableAFComputeProvider.estimate

        @wraps(original)
        def observed(provider, query):
            self.assertIs(type(query), AFComputeQuery)
            return original(provider, query)

        with patch.object(OperatorTableAFComputeProvider, "estimate", observed):
            provider = self.compute()
            provider.estimate(self.prepared(provider, _compute_query()))

    def test_configuration_snapshot_rejects_updates_and_detaches_source(self):
        config = configuration()
        planner = ParallelExecutionPlanner(config.cluster, runtime_plan=config.plan)
        original = planner.cluster_config.af.router_location
        config.cluster.af.router_location = "f" if original == "a" else "a"
        self.assertEqual(planner.cluster_config.af.router_location, original)
        with self.assertRaisesRegex(TypeError, "new runtime"):
            planner.cluster_config.af.router_location = "f"
        with self.assertRaisesRegex(TypeError, "new runtime"):
            planner.cluster_config = config.cluster
        with self.assertRaises(TypeError):
            planner.cluster_config.networks["new"] = "changed"
        provider = self.compute(lookup_policy=LookupPolicy(extrapolate="none"))
        for name, value in (
            ("lookup_policy", LookupPolicy()),
            ("fallback", "table_only"),
            ("hardware", object()),
        ):
            with self.assertRaisesRegex(TypeError, "new runtime"):
                setattr(provider, name, value)

    def test_dataset_is_pinned_on_first_use_and_new_provider_sees_update(self):
        hardware = af_test_hardware(moe=_moe())
        source = SourceRecord("measured:test", "A", "measured")

        def package(version, latency):
            return OperatorDataPackage.from_rows(
                PackageMeta(
                    version, "a-device", "measured", sources={source.source_id: source}
                ),
                {
                    "gemm": [
                        dict(
                            dtype="fp16",
                            m=2,
                            n=48,
                            k=16,
                            latency_us=latency,
                            source_id=source.source_id,
                        )
                    ]
                },
            )

        first = package("v1", 123.0)
        hardware.register_package(first)
        provider = OperatorTableAFComputeProvider(hardware, hardware.model("TestModel"))
        query = _compute_query()
        initial = provider.estimate(self.prepared(provider, query))
        first.tables["gemm"][0]["latency_us"] = 999.0
        hardware.register_package(package("v2", 456.0))
        provider._estimate_cache.clear()
        self.assertEqual(provider.estimate(self.prepared(provider, query)), initial)
        pinned = provider.resolver_for("a-device").package
        with self.assertRaises(TypeError):
            pinned.tables["gemm"][0]["latency_us"] = 999.0
        updated = OperatorTableAFComputeProvider(hardware, hardware.model("TestModel"))
        estimate = updated.estimate(self.prepared(updated, query))
        self.assertEqual(estimate.dataset_version, "v2")
        self.assertNotEqual(estimate, initial)

    def test_fast_full_and_uncached_paths_have_identical_service_records(self):
        original = OperatorTableAFComputeProvider.__init__
        original_transfer = TopologyAFTransferProvider.__init__
        for trace in (False, True):
            results = []
            for mode in ("fast", "full", "uncached"):

                def initialize(provider, *args, **kwargs):
                    original(provider, *args, **kwargs)
                    provider.query_fast_path = mode == "fast"

                def initialize_transfer(provider, *args, **kwargs):
                    original_transfer(provider, *args, **kwargs)
                    provider.query_fast_path = mode == "fast"

                config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
                config.cluster.af.service_estimate_cache = mode != "uncached"
                with (
                    patch.object(
                        OperatorTableAFComputeProvider, "__init__", initialize
                    ),
                    patch.object(
                        TopologyAFTransferProvider, "__init__", initialize_transfer
                    ),
                    observe_services() as services,
                ):
                    executor, rows, compute, transfer = simulate(
                        config, 2, 8, steps=2, trace=trace, fallback="analytical_only"
                    )
                results.append((rows, services, executor.trace.events))
                for provider in (compute, transfer):
                    expected = (
                        provider.estimate_requests - provider.estimate_cache_hits
                        if mode == "fast"
                        else provider.estimate_requests
                    )
                    self.assertEqual(provider.query_constructions, expected)
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])


class CacheNamespaceTest(unittest.TestCase):
    def test_equal_hashes_do_not_alias_distinct_configuration_values(self):
        # Python deliberately gives these unequal integers the same hash.
        left, right = EstimateNamespace((-1,)), EstimateNamespace((-2,))
        self.assertEqual(hash(left), hash(right))
        self.assertNotEqual(left, right)
        cache = {left: "left", right: "right"}
        self.assertEqual(cache[EstimateNamespace((-1,))], "left")
        self.assertEqual(cache[EstimateNamespace((-2,))], "right")

    def test_extension_keeps_previous_namespace_immutable_and_recomputes_hash(self):
        original = EstimateNamespace(("fallback", ("scope", "a")))
        extended = original + (("profile", "new"),)
        self.assertEqual(original.values, ("fallback", ("scope", "a")))
        self.assertEqual(extended.values, original.values + (("profile", "new"),))
        self.assertEqual(hash(extended), hash(extended.values))
        self.assertEqual(extended, EstimateNamespace(extended.values))
        with self.assertRaises(FrozenInstanceError):
            original.digest = 0
        with self.assertRaises(FrozenInstanceError):
            original.values = ()

    def test_service_cache_checks_full_namespace_after_a_hash_collision(self):
        provider = CacheFastPathTest().compute()
        provider._estimate_namespace = EstimateNamespace((-1,))
        query = _compute_query()
        first = provider.estimate(query)
        self.assertIs(provider.estimate(query), first)
        hits = provider.estimate_cache_hits
        provider._estimate_namespace = EstimateNamespace((-2,))
        self.assertIsNot(provider.estimate(query), first)
        self.assertEqual(provider.estimate_cache_hits, hits)
