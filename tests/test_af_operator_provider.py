"""AF latency providers built on the operator-table stack."""

from __future__ import annotations

import unittest
from dataclasses import replace

import simpy

from tests.test_af_executor import (
    _cluster_dict,
    _psla_config,
    af_test_hardware,
)
from TokenSim.af import (
    AF_MATCH_TYPES,
    AFComputeQuery,
    AFTransferQuery,
    OperatorTableAFComputeProvider,
    TopologyAFTransferProvider,
    resolve_af_latency_providers,
)
from TokenSim.config.cluster_config import AFConfig, ClusterConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, reset_g_time
from TokenSim.moe.config import MoEModelConfig
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord
from util.request import LLMSource
from util.results import get_af_stats

HIDDEN = 16


def _moe():
    return MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=2,
        num_moe_layers=1,
        moe_intermediate_size=32,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
    )


def _compute_query(
    stage="a",
    *,
    phase="decode",
    tokens=2,
    context=(2,),
    tp_size=1,
    ep_size=1,
    expert_workload=None,
    dtype="fp16",
    local_shape=None,
):
    if expert_workload is None and stage == "expert":
        expert_workload = ((0, tokens),)
    if local_shape is None:
        local_shape = {
            "tokens": tokens,
            "hidden_size": HIDDEN if stage == "router" else -(-HIDDEN // tp_size),
            "global_hidden_size": HIDDEN,
            "tp_size": tp_size,
        }
    return AFComputeQuery(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase=phase,
        stage=stage,
        token_count=tokens,
        context_lengths=context,
        model_id="TestModel",
        dtype=dtype,
        device_type="a-device" if stage in {"a", "router"} else "f-device",
        worker_id=0 if stage in {"a", "router"} else 1,
        operation={
            "a": "a_rank_compute",
            "router": "router_topk",
            "expert": "expert_rank_compute",
        }[stage],
        source_domain="a-domain" if stage in {"a", "router"} else "f-domain",
        logical_participant="a0" if stage in {"a", "router"} else "f0",
        physical_worker="a-pool:0" if stage in {"a", "router"} else "f-pool:0",
        local_shape=local_shape,
        local_token_count=tokens,
        expert_ids=tuple(expert_id for expert_id, _ in (expert_workload or ())),
        expert_workload=tuple(expert_workload or ()),
        tp_group=tuple(range(tp_size)),
        ep_group=tuple(range(ep_size)) if stage == "expert" else (),
    )


def _transfer_query(
    operation="af_a_to_f",
    *,
    payload_bytes=4096,
    endpoints=("a-pool:0", "f-pool:0"),
    network="af-net",
    group_size=None,
):
    direction = {
        "af_a_to_f": "a_to_f",
        "af_f_to_a": "f_to_a",
        "tp_collective": "tp_collective",
        "ep_dispatch": "a_to_f",
        "ep_combine": "f_to_a",
    }.get(operation, "a_to_f")
    return AFTransferQuery(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase="decode",
        direction=direction,
        token_count=2,
        hidden_size=HIDDEN,
        dtype="fp16",
        source_device="a-device",
        target_device="f-device",
        network=network,
        operation=operation,
        group_size=group_size,
        payload_bytes=payload_bytes,
        physical_endpoints=endpoints,
        participant_ranks=tuple(range(len(endpoints))),
    )


def _component_names(estimate):
    return [item["name"] for item in estimate.details["components"]]


class OperatorTableComputeProviderTest(unittest.TestCase):
    def setUp(self):
        self.hardware = af_test_hardware(moe=_moe())
        self.model = self.hardware.model("TestModel")
        self.provider = OperatorTableAFComputeProvider(self.hardware, self.model)

    def test_stage_decomposition_matches_the_af_contract(self):
        a = self.provider.estimate(_compute_query("a"))
        router = self.provider.estimate(_compute_query("router"))
        expert = self.provider.estimate(_compute_query("expert"))
        self.assertEqual(
            _component_names(a),
            ["a_rmsnorm", "qkv_projection", "rope", "attention_core", "output_projection", "a_residual_add"],
        )
        self.assertEqual(_component_names(router), ["f_rmsnorm", "router"])
        self.assertEqual(_component_names(expert), ["moe_expert", "f_residual_add"])
        for estimate in (a, router, expert):
            total = sum(item["latency_us"] for item in estimate.details["components"])
            self.assertAlmostEqual(estimate.service_duration, total / 1e6)
            self.assertGreater(estimate.service_duration, 0)
            # No measured package for the test devices: every operator is analytical.
            self.assertEqual(estimate.match_type, "analytical")
            self.assertIn(estimate.match_type, AF_MATCH_TYPES)
        self.assertEqual(a.dataset_version, "analytical:a-device")
        self.assertEqual(expert.dataset_version, "analytical:f-device")
        self.assertEqual(a.details["operator_backend"], "analytical")
        self.assertGreater(self.provider.stats_dict()["operator_query_count"], 0)

    def test_tensor_parallel_shrinks_rank_local_projections(self):
        a = self.provider.estimate(_compute_query("a", tp_size=2))
        keys = {item["name"]: item["key"] for item in a.details["components"]}
        # hidden 16, 8 heads of dim 2 -> q_dim 16, kv_dim 16; TP 2 halves both.
        self.assertEqual((keys["qkv_projection"]["n"], keys["qkv_projection"]["k"]), (24, HIDDEN))
        self.assertEqual((keys["output_projection"]["n"], keys["output_projection"]["k"]), (HIDDEN, 8))
        self.assertEqual(keys["attention_core"]["num_heads"], 4)
        self.assertEqual(keys["attention_core"]["num_kv_heads"], 4)
        self.assertEqual(keys["rope"]["hidden_size"], 16)

    def test_prefill_and_decode_group_attention_by_request_length(self):
        prefill = self.provider.estimate(_compute_query("a", phase="prefill", tokens=5, context=(2, 3)))
        attention = [item for item in prefill.details["components"] if item["name"] == "attention_core"]
        self.assertEqual([item["table"] for item in attention], ["context_attention"] * 2)
        self.assertEqual(
            [(item["key"]["input_seq_len"], item["key"]["batch_size"]) for item in attention],
            [(2, 1), (3, 1)],
        )
        decode = self.provider.estimate(_compute_query("a", tokens=3, context=(4, 8, 4)))
        attention = [item for item in decode.details["components"] if item["name"] == "attention_core"]
        self.assertEqual([item["table"] for item in attention], ["generation_attention"] * 2)
        self.assertEqual(
            [(item["key"]["context_len"], item["key"]["batch_size"]) for item in attention],
            [(4, 2), (8, 1)],
        )

    def test_moe_expert_uses_balanced_equivalent_tokens(self):
        expert = self.provider.estimate(
            _compute_query("expert", tokens=6, ep_size=2, expert_workload=((0, 4), (1, 2)))
        )
        moe = next(item for item in expert.details["components"] if item["name"] == "moe_expert")
        # 6 routed assignments on one of 2 EP ranks with top_k 2 -> 6 balanced input tokens.
        self.assertEqual(moe["key"]["num_tokens"], 6)
        self.assertEqual(moe["key"]["routed_assignments"], 6)
        self.assertEqual(moe["key"]["per_expert"], ((0, 4), (1, 2)))
        self.assertEqual((moe["key"]["ep_size"], moe["key"]["top_k"], moe["key"]["num_experts"]), (2, 2, 4))
        self.assertEqual(moe["key"]["inter_size"], 32)

    def test_dense_model_prices_ffn_gemms_on_the_f_rank(self):
        hardware = af_test_hardware()
        provider = OperatorTableAFComputeProvider(hardware, hardware.model("TestModel"))
        expert = provider.estimate(_compute_query("expert", tokens=2, tp_size=2, expert_workload=()))
        self.assertEqual(
            _component_names(expert),
            ["dense_gate_up", "dense_swiglu", "dense_down", "f_residual_add"],
        )
        keys = {item["name"]: item["key"] for item in expert.details["components"]}
        self.assertEqual((keys["dense_gate_up"]["n"], keys["dense_gate_up"]["k"]), (64, HIDDEN))
        self.assertEqual((keys["dense_down"]["n"], keys["dense_down"]["k"]), (HIDDEN, 32))
        router = provider.estimate(_compute_query("router"))
        self.assertEqual(_component_names(router), ["f_rmsnorm"])

    def test_invalid_local_work_and_expert_metadata_fail(self):
        with self.assertRaisesRegex(ValueError, "local_shape fields"):
            self.provider.estimate(_compute_query("a", local_shape={"tokens": 2}))
        with self.assertRaisesRegex(ValueError, "does not match model hidden_size"):
            self.provider.estimate(
                _compute_query(
                    "a",
                    local_shape={"tokens": 2, "hidden_size": 8, "global_hidden_size": 8, "tp_size": 1},
                )
            )
        with self.assertRaisesRegex(ValueError, "does not match local_token_count"):
            self.provider.estimate(_compute_query("expert", tokens=3, expert_workload=((0, 2),)))
        with self.assertRaisesRegex(ValueError, "unsupported AF compute stage"):
            self.provider.estimate(replace(_compute_query("a"), stage="other"))

    def test_measured_rows_take_precedence_and_table_only_rejects_gaps(self):
        source = SourceRecord("measured:test", "A", "measured")
        meta = PackageMeta("test-v1", "a-device", "measured", sources={"measured:test": source})
        package = OperatorDataPackage.from_rows(
            meta,
            {"gemm": [{"dtype": "fp16", "m": 2, "n": 48, "k": HIDDEN, "latency_us": 123.0, "source_id": "measured:test"}]},
        )
        self.hardware.register_package(package)
        provider = OperatorTableAFComputeProvider(self.hardware, self.model)
        a = provider.estimate(_compute_query("a"))
        qkv = next(item for item in a.details["components"] if item["name"] == "qkv_projection")
        self.assertEqual((qkv["match_type"], qkv["latency_us"], qkv["source_id"]), ("exact", 123.0, "measured:test"))
        self.assertEqual(a.dataset_version, "test-v1")
        # the remaining operators are still analytical, which is the composition's grade
        self.assertEqual(a.match_type, "analytical")
        self.assertTrue(a.source_id.startswith("compose:"))
        strict = OperatorTableAFComputeProvider(self.hardware, self.model, fallback="table_only")
        with self.assertRaises(MissingOperatorDataError):
            strict.estimate(_compute_query("a"))
        with self.assertRaises(ConfigurationError):
            OperatorTableAFComputeProvider(self.hardware, self.model, fallback="table_only").resolver_for("f-device")


class TopologyTransferProviderTest(unittest.TestCase):
    def setUp(self):
        self.cluster = ClusterConfig(**_cluster_dict(attention_workers=2, ffn_workers=2))
        self.hardware = af_test_hardware()
        self.workers = self.cluster.workers()
        self.placement = build_topology_placement(self.cluster, self.workers, self.hardware)
        self.provider = TopologyAFTransferProvider(
            self.hardware, self.cluster, self.workers, self.placement, dtype="fp16"
        )

    def test_cross_domain_transfer_uses_the_link_class(self):
        link = self.hardware.link("profile")
        estimate = self.provider.estimate(_transfer_query("af_a_to_f", payload_bytes=4096))
        self.assertAlmostEqual(estimate.service_duration, link.transfer_us(4096) / 1e6)
        self.assertEqual(estimate.payload_bytes, 4096)
        self.assertEqual(estimate.match_type, "analytical")
        self.assertEqual(estimate.details["link"], "nvlink-test")
        self.assertEqual(estimate.dataset_version, "topology:synth_af-net")
        back = self.provider.estimate(_transfer_query("af_f_to_a", payload_bytes=4096))
        self.assertEqual(back.service_duration, estimate.service_duration)
        self.assertEqual(self.provider.estimate(_transfer_query(payload_bytes=0)).service_duration, 0.0)

    def test_tp_collective_uses_the_collective_model_or_a_ring_fallback(self):
        query = _transfer_query("tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1"))
        placed = self.provider.estimate(query)
        self.assertEqual(placed.details["provider"], "collective_model")
        self.assertGreater(placed.service_duration, 0)
        self.assertIn(placed.match_type, AF_MATCH_TYPES)
        with self.assertRaisesRegex(ConfigurationError, "requires placement"):
            TopologyAFTransferProvider(self.hardware, self.cluster, self.workers, None)
        values = _cluster_dict(attention_workers=2, ffn_workers=2)
        values["af"]["transfer_topology_mode"] = "flat_link"
        unplaced = TopologyAFTransferProvider(self.hardware, ClusterConfig(**values), self.workers, None)
        ring = unplaced.estimate(query)
        self.assertEqual(ring.details["algorithm"], "ring")
        self.assertGreater(ring.service_duration, 0)
        self.assertEqual(
            self.provider.estimate(replace(query, payload_bytes=0)).service_duration, 0.0
        )

    def test_unknown_endpoint_cannot_silently_use_flat_link(self):
        for payload in (0, 8192):
            with self.subTest(payload=payload), self.assertRaisesRegex(ConfigurationError, "known physical endpoints"):
                self.provider.estimate(_transfer_query(payload_bytes=payload, endpoints=("a-pool:0", "missing:0")))

    def test_physical_endpoint_paths_distinguish_node_rack_and_cluster(self):
        from TokenSim.hardware.topology import TopologyLevel, TopologyPlacement, TopologySpec
        topology = TopologySpec("three-level-af", (
            TopologyLevel("node", 4, "nvlink-test", links_per_endpoint=2),
            TopologyLevel("rack", 2, "ethernet-test"),
            TopologyLevel("cluster", None, "ethernet-test", oversubscription=2),
        ))
        times = []
        for index, level in ((1, 0), (4, 1), (8, 2)):
            placement = TopologyPlacement(topology, {0: 0, 1: 2, 2: index, 3: 16})
            provider = TopologyAFTransferProvider(self.hardware, self.cluster, self.workers, placement)
            estimate = provider.estimate(_transfer_query(payload_bytes=1 << 20))
            self.assertEqual(estimate.details["lowest_common_level"], level)
            times.append(estimate.service_duration)
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])

    def test_separated_collective_removes_software_launch_cost(self):
        from TokenSim.config.cluster_config import AFCommunicationTimingConfig
        query = _transfer_query("tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1"))
        legacy = self.provider.estimate(query)
        self.cluster.af.communication_timing = AFCommunicationTimingConfig(enabled=True, host_submit_us=5)
        # A provider pins its configuration; reconfiguration creates a new one.
        self.assertIs(self.provider.estimate(query), legacy)
        separated = TopologyAFTransferProvider(
            self.hardware, self.cluster, self.workers, self.placement, dtype="fp16"
        ).estimate(query)
        self.assertEqual(separated.details["launch_us"], 0)
        self.assertEqual(separated.details["timing_scope"], "transport_service")
        self.assertAlmostEqual(legacy.service_duration - separated.service_duration,
                               legacy.details["launch_us"] / 1e6)

    def test_measured_collective_retains_declared_or_unspecified_scope(self):
        from unittest.mock import patch
        from TokenSim.comm.collectives import CollectiveEstimate
        model = self.provider._collective_model_for("a-pool:0")
        query = _transfer_query("tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1"))
        measured = CollectiveEstimate(10, "all_reduce", 2, 8192, "measured", "fixture:measured")
        self.provider.cache_enabled = False
        with patch.object(model, "estimate", return_value=measured):
            for scope in ("legacy_unspecified", "eager_interval", "transport_service"):
                model.timing_scope = scope
                model.dataset_version = "measured-fixture-v1"
                estimate = self.provider.estimate(query)
                self.assertEqual(estimate.details["timing_scope"], scope)
                self.assertEqual(estimate.dataset_version, "measured-fixture-v1")

    def test_unknown_network_or_operation_fail(self):
        values = _cluster_dict()
        values["networks"] = {"af-net": "no-such-link"}
        broken = ClusterConfig(**values)
        with self.assertRaisesRegex(ConfigurationError, "not a link class"):
            TopologyAFTransferProvider(self.hardware, broken, broken.workers(), None)
        with self.assertRaisesRegex(ValueError, "unsupported AF communication operation"):
            self.provider.estimate(replace(_transfer_query(), operation="broadcast"))
        with self.assertRaisesRegex(ValueError, "requires explicit payload_bytes"):
            self.provider.estimate(replace(_transfer_query(), payload_bytes=None))
        with self.assertRaisesRegex(ConfigurationError, "not declared"):
            self.provider.estimate(_transfer_query(network="ghost-net"))


class ProviderSelectionTest(unittest.TestCase):
    def test_resolve_af_latency_providers_modes(self):
        injected = AFConfig(enabled=True, latency_provider="injected")
        compute, transfer = object(), object()
        self.assertEqual(resolve_af_latency_providers(injected, "x.json", compute, transfer), (compute, transfer))
        table = AFConfig(enabled=True, latency_provider="operator_table")
        self.assertEqual(resolve_af_latency_providers(table, "x.json"), (None, None))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_af_latency_providers(table, "x.json", compute, transfer)

    def test_config_rejects_unknown_provider_and_fallback(self):
        with self.assertRaises(ConfigurationError):
            AFConfig(latency_provider="operator_data")
        with self.assertRaises(ConfigurationError):
            AFConfig(latency_fallback="bogus")
        with self.assertRaises(ConfigurationError):
            AFConfig(operator_backend="")
        config = AFConfig(latency_provider="operator_table", operator_backend="vllm", latency_fallback="table_only")
        self.assertEqual((config.operator_backend, config.latency_fallback), ("vllm", "table_only"))

    def test_engine_builds_operator_table_providers_and_runs_prefill_and_decode(self):
        reset_g_time()
        values = _cluster_dict()
        values["af"]["latency_provider"] = "operator_table"
        cluster = ClusterConfig(**values)
        hardware = af_test_hardware(moe=_moe())
        psla = replace(
            _psla_config(),
            is_moe_model=True,
            num_experts=4,
            num_experts_per_tok=2,
            num_moe_layers=1,
            moe_intermediate_size=32,
        )
        env = simpy.Environment()
        engine = LLMEngine(
            env=env,
            block_size=4,
            batching="dynamic",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=psla,
            cluster_config=cluster,
            hardware=hardware,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            af_dtype="fp16",
        )
        self.assertIsInstance(engine.af_compute_provider, OperatorTableAFComputeProvider)
        self.assertIsInstance(engine.af_transfer_provider, TopologyAFTransferProvider)
        request = Request(id=7, prefill_len=8, decode_len=2, block_size=4)
        env.process(LLMSource(env=env, engine=engine, requests=[request], qps=1, distribution="burst"))

        def stop_when_done():
            while env.now < 1.0:
                if request.is_done:
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-9)
                    return
                yield env.timeout(1e-6)
            raise AssertionError("operator-table AF request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        self.assertTrue(request.is_done)
        self.assertEqual([summary["phase"] for summary in engine.af_iteration_summaries], ["prefill", "decode"])
        stats = get_af_stats(engine)
        events = [event for summary in stats["af_trace"] for event in summary["events"]]
        self.assertTrue(events)
        self.assertTrue(all(event["match_type"] in AF_MATCH_TYPES for event in events))
        compute_events = [event for event in events if event["stage"] in {"a", "router", "expert"}]
        self.assertTrue(all(event["dataset_version"].startswith("analytical:") for event in compute_events))
        self.assertTrue(all(event["service_duration"] > 0 for event in compute_events))
        self.assertEqual(stats["af_latency_providers"]["compute"]["provider"], "operator_table")
        self.assertEqual(stats["af_latency_providers"]["transfer"]["provider"], "topology_links")
        self.assertGreater(stats["af_operator_stats"]["operator_query_count"], 0)


if __name__ == "__main__":
    unittest.main()
