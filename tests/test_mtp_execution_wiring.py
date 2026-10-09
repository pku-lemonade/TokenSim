"""Complete worker wiring for MTP K0 and enabled AL1; no workload timing run."""

import json
from pathlib import Path
from unittest import TestCase

import simpy
import yaml

from TokenSim.config.config import ClusterConfig
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.config.speculative_execution import resolve_mtp_tokens
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.llm.forward import ForwardPass, RankShape
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from scripts.run_agentx_native import command


class MTPExecutionWiringTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load("data")
        cls.model = cls.hardware.model("DeepSeek-V4-Pro")
        cls.psla = PSLAConfig.from_file("data/psla/deepseek-v4-pro.json")
        cls.engines = {}
        for sku in ("b200", "b300"):
            for k, acceptance in ((0, 1.0), (3, 1.0), (3, 2.49), (None, 1.0), (None, 2.49)):
                cluster = ClusterConfig.from_file(f"data/clusters/8_{sku}/dep8.json")
                cls.engines[sku, k, acceptance] = LLMEngine(
                    env=simpy.Environment(), block_size=256, batching="paged-attn",
                    kv_transfer_config=cluster.effective_kv_transfer(), psla_config=cls.psla,
                    cluster_config=cluster, parallel_config=cluster.effective_parallel_config(),
                    hardware=cls.hardware, prefill_worker_pool_type="round_robin",
                    decode_worker_pool_type="round_robin", max_parallem_sum=49 if sku == "b200" else 96,
                    max_num_batched_tokens=8192 if sku == "b200" else 16384,
                    gpu_memory_utilization=.9 if sku == "b200" else .92,
                    router_policy="consistent_hash", dp_lockstep=True, dp_sync_interval=16,
                    long_prefill_token_threshold=512, prefill_schedule_interval=8,
                    latency_backend_type="operator_table", latency_fallback="table_first", operator_backend="vllm",
                    moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
                    attention_backend="module", mhc_backend="module",
                    mtp_acceptance_length=acceptance, mtp_num_speculative_tokens=k,
                )

    def test_disabled_all_eight_workers_use_q1_no_draft_or_lookahead(self):
        for sku in ("b200", "b300"):
            engine = self.engines[sku, 0, 1.0]
            self.assertEqual(engine.env.now, 0)
            for worker in engine.workers:
                with self.subTest(sku=sku, rank=worker.dp_rank):
                    self.assertFalse(worker.scheduler.mtp_enabled)
                    self.assertEqual(worker.mtp_num_speculative_tokens, 0)
                    self.assertEqual(worker.cache_config.hybrid_layout.speculative_tokens, 0)
                    self.assertEqual(worker.latency_backend.mtp_query_width, 1)
                    self.assertIsNone(worker.latency_backend.native_mtp)
                    self.assertEqual(engine.dp_coordinator.dummies[worker.id].scheduled_query_tokens, 1)
                    request = Request(1, 8192, 20, 256)
                    request.generation_idx = 1
                    request.num_computed_tokens = 8192
                    self.assertEqual(worker.scheduler._query_tokens(request, 1), 1)
                    self.assertEqual(worker.scheduler._decode_lookahead(request), 0)

    def test_disabled_weights_and_kv_remove_actual_mtp_module(self):
        for sku in ("b200", "b300"):
            engine = self.engines[sku, 0, 1.0]
            for worker in engine.workers:
                with self.subTest(sku=sku, rank=worker.dp_rank):
                    cache = worker.cache_config
                    self.assertFalse(any(row["name"].startswith("mtp.") for row in cache.weight_memory_report["tensors"]))
                    self.assertFalse(any(layer.draft for group in cache.hybrid_layout.groups for layer in group.layers))
                    self.assertFalse(any(group.eagle for group in cache.hybrid_layout.groups))
                    persistent = cache.memory_budget["persistent_components"]
                    for name in ("target_mtp_hidden", "drafter_hidden", "draft_topk", "swa_rope_cache"):
                        self.assertEqual(persistent[name], 0)
                    self.assertEqual(cache.memory_budget["graph_storage"]["draft_prefill_descriptors"], 0)
                    self.assertEqual(cache.memory_budget["graph_storage"]["draft_decode_descriptors"], 0)

    def test_enabled_al1_retains_verify_width_draft_and_lookahead(self):
        for sku in ("b200", "b300"):
            engine = self.engines[sku, 3, 1.0]
            for worker in engine.workers:
                with self.subTest(sku=sku, rank=worker.dp_rank):
                    self.assertTrue(worker.scheduler.mtp_enabled)
                    self.assertEqual(worker.latency_backend.mtp_query_width, 4)
                    self.assertEqual(worker.latency_backend.native_mtp.k, 3)
                    self.assertEqual(engine.dp_coordinator.dummies[worker.id].scheduled_query_tokens, 4)
                    request = Request(1, 8192, 20, 256)
                    request.generation_idx = 1
                    request.num_computed_tokens = 8192
                    self.assertEqual(worker.scheduler._decode_output_tokens(request, 19), 1)
                    self.assertEqual(worker.scheduler._query_tokens(request, 1), 4)
                    self.assertEqual(worker.scheduler._decode_lookahead(request), 3)

    def test_reference_k3_cache_budget_is_identical_to_frozen_prior_stage(self):
        launch = json.loads(Path("data/reference/agentx-native-memory/launch-budget-audit.json").read_text())
        for sku in ("b200", "b300"):
            for worker in self.engines[sku, 3, 2.49].workers:
                with self.subTest(sku=sku, rank=worker.dp_rank):
                    self.assertEqual(worker.cache_config.memory_budget, launch["hardware"][sku]["budget"])
                    self.assertEqual(worker.cache_config.num_gpu_blocks, launch["hardware"][sku]["pool_slots"])
                    self.assertTrue(worker.scheduler.mtp_enabled)
                    self.assertEqual(worker.latency_backend.native_mtp.k, 3)

    def test_implicit_reference_and_disabled_defaults_match_explicit_configuration(self):
        for sku in ("b200", "b300"):
            for k, acceptance in ((0, 1.0), (3, 2.49)):
                first = self.engines[sku, k, acceptance].workers[0]
                second = self.engines[sku, None, acceptance].workers[0]
                self.assertEqual(first.cache_config.to_dict(), second.cache_config.to_dict())
                self.assertEqual(first.scheduler.mtp_enabled, second.scheduler.mtp_enabled)
                self.assertEqual(first.latency_backend.mtp_query_width, second.latency_backend.mtp_query_width)

    def test_disabled_estimate_has_only_main_moe_pass_and_no_draft_component(self):
        for sku in ("b200", "b300"):
            backend = self.engines[sku, 0, 1.0].workers[0].latency_backend
            backend.reset_profile_stats()
            backend.set_moe_step(ForwardPass({rank: RankShape(1, 1, False, 1) for rank in range(8)}))
            request = Request(1, 8192, 20, 256)
            request.generation_idx = 1
            request.num_computed_tokens = 8192
            request.scheduled_tokens = request.scheduled_query_tokens = 1
            self.assertGreater(backend.estimate_step_latency([request]), 0)
            self.assertEqual(backend.native_moe.counts["rank_estimates"], 1)
            self.assertNotIn("speculative_counts", backend.stats_dict())
            self.assertIsNone(backend.native_mtp)

    def test_runner_forwards_explicit_k0_and_default_k3(self):
        config = yaml.safe_load(Path("data/experiments/agentx.yaml").read_text())
        for sku in ("b200", "b300"):
            for suffix, expected in (("", "3"), ("-nomtp", "0")):
                concurrency = 196 if sku == "b200" else 384
                point = next(row for row in config["experiments"]
                             if row["id"] == f"{sku}-c{concurrency}-dep8-simplecpu{suffix}")
                argv = command(point, config["defaults"], Path("/unused-result"), Path("/unused-connector"))
                self.assertEqual(argv[argv.index("--mtp_num_speculative_tokens") + 1], expected)

    def test_explicit_invalid_k_and_acceptance_are_rejected(self):
        for k, acceptance in ((-1, 1.0), (True, 1.0), (1.5, 1.0), (0, 2.49), (1, 2.49), (3, 0.5)):
            with self.subTest(k=k, acceptance=acceptance), self.assertRaises(ConfigurationError):
                resolve_mtp_tokens(self.model, acceptance, k)
        self.assertEqual(resolve_mtp_tokens(self.model, 1.0, 1), 1)
