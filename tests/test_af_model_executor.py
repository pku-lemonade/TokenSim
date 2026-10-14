import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.model_executor import ModelAFExecutor
from TokenSim.af.planning import RankLocalWork
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware import HardwareContext
from TokenSim.llm.llm_engine import LLMEngine
from scripts.deepseek_v3_runtime import configuration, simulate
from tests.af_runtime_oracle import observe_services
from tests.support.af_engine import _run_request


class ModelExecutorTest(unittest.TestCase):
    def test_dense_mha_model_can_use_whole_model_executor(self):
        from tests.support.hardware import make_model

        config = configuration()
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=make_model(num_layers=1)
        )
        executor, _, _, _ = simulate_decode(config, 1, 8, fallback="analytical_only")
        self.assertTrue(any(e.stage == "dense_ffn" for e in executor.trace.events))
        self.assertTrue(any(e.operation == "lm_head" for e in executor.trace.events))

    def test_script_trace_off_skips_boundaries_and_matches_services(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        results = []
        for trace in (True, False):
            with observe_services() as services:
                if trace:
                    executor, rows, _, _ = simulate(
                        config, 2, 8, steps=2, trace=trace, fallback="analytical_only"
                    )
                else:
                    with (
                        patch(
                            "TokenSim.af.executor_compute.AFTraceEvent",
                            side_effect=AssertionError("trace constructed"),
                        ),
                        patch.object(
                            RankLocalWork,
                            "as_dict",
                            side_effect=AssertionError("local work serialized"),
                        ),
                    ):
                        executor, rows, _, _ = simulate(
                            config,
                            2,
                            8,
                            steps=2,
                            trace=trace,
                            fallback="analytical_only",
                        )
            results.append((rows, services))
            self.assertEqual(bool(executor.trace.events), trace)
        self.assertEqual(results[0], results[1])

    def test_boundaries_use_declared_lanes_and_separated_timing(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        config.doc["af"]["transfer_topology_mode"] = "flat_link"
        config.doc["af"]["communication_timing"] = dict(
            enabled=True, host_submit_us=5, host_completion_us=2
        )
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=config.model
        )
        executor, _, _, _ = simulate_decode(config, 2, 8, fallback="analytical_only")
        events = [e for e in executor.trace.events if e.layer_id == -1]
        collectives = [e for e in events if e.event_type == "collective"]
        self.assertEqual(
            {e.operation: e.resource_id for e in collectives},
            {
                "all_reduce": "link:af-fabric:a_to_f",
                "all_gather": "link:af-fabric:f_to_a",
            },
        )
        self.assertTrue(any(e.event_type == "host_submit" for e in events))
        self.assertTrue(any(e.event_type == "synchronize" for e in events))
        executor.trace.validate_invariants(0)

    def test_prefill_restrictions_fail_before_iteration_creation(self):
        executor, _, _, _ = simulate(configuration(), 1, 8, fallback="analytical_only")
        for requests in (
            [
                SimpleNamespace(
                    id=0,
                    is_prefill=True,
                    prefill_len=8,
                    prefill_compute_len=4,
                    context_len=8,
                )
            ],
            [
                SimpleNamespace(
                    id=i,
                    is_prefill=True,
                    prefill_len=n,
                    prefill_compute_len=n,
                    context_len=n,
                )
                for i, n in enumerate((4, 8))
            ],
        ):
            previous = executor._next_iteration_id
            with self.assertRaisesRegex(ConfigurationError, "equal-length"):
                executor.submit(0, requests, "prefill")
            self.assertEqual(executor._next_iteration_id, previous)
            self.assertFalse(executor.active_iterations)

    def test_f_router_metadata_uses_endpoint_provider(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=2)
        config.doc["af"]["router_location"] = "f"
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=config.model
        )
        executor, _, _, _ = simulate_decode(config, 2, 8, fallback="analytical_only")
        metadata = [
            event
            for event in executor.trace.events
            if event.operation == "af_router_metadata"
        ]
        self.assertTrue(metadata)
        self.assertTrue(
            all(
                event.details["timing_scope"] == "transport_service"
                for event in metadata
            )
        )
        executor.trace.validate_invariants(0)

    def test_collective_hosts_submit_independently_under_contention(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        config.doc["af"]["communication_timing"] = dict(
            enabled=True, host_submit_us=5, host_completion_us=2
        )
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=config.model
        )
        executor, _, _, _ = simulate_decode(config, 1, 8, fallback="analytical_only")
        executor.trace.clear()
        env = executor.env
        start = env.now
        block = SimpleNamespace(
            iteration_id=99, block_id=0, phase="decode", request_count=1, token_count=1
        )
        step = SimpleNamespace(
            operation="all_reduce",
            direction="collective",
            logical_stage_id=None,
            leg_id=None,
            source_domain="a-domain",
            target_domain="a-domain",
            layer_type=None,
        )
        host = executor.resources.host_resources_for(("a-pool:1",))[0][1]
        held = host.request()

        def release():
            yield env.timeout(10e-6)
            host.release(held)

        env.process(release())
        env.process(
            executor._run_communication_host_stage(
                block,
                -1,
                step,
                "test-submit",
                (),
                ("a-pool:0", "a-pool:1"),
                5e-6,
                "host_submit",
            )
        )
        env.run()
        events = [
            event
            for event in executor.trace.events
            if event.event_type == "host_submit"
        ]
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].start_at, start)
        self.assertAlmostEqual(events[0].end_at, start + 5e-6)
        self.assertAlmostEqual(events[1].start_at, start + 10e-6)
        self.assertAlmostEqual(env.now, start + 15e-6)

    def test_boundary_failure_cancels_other_blocks_and_releases_resources(self):
        executor, _, _, _ = simulate(
            configuration(a_tp=2, f_tp=2, af_batch_size=1),
            1,
            8,
            fallback="analytical_only",
        )
        original = executor.compute_provider.estimate

        def fail(query):
            if query.operation == "lm_head" and query.block_id == 0:
                raise ValueError("missing boundary shape")
            return original(query)

        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
            )
            for i in range(3)
        ]
        iteration = executor.submit(0, requests, "decode")
        with patch.object(executor.compute_provider, "estimate", fail):
            executor.env.process(executor.run_iteration(iteration))
            with self.assertRaisesRegex(
                SimulationStateError, "lm_head.*missing boundary shape"
            ):
                executor.env.run()
        self.assertIsNone(iteration.end_at)
        self.assertFalse(executor.active_iterations)
        for resource in (
            *executor.resources.compute_slots.values(),
            *executor.resources.communication_resources.values(),
        ):
            self.assertEqual(resource.count, 0)
            self.assertFalse(resource.queue)

    def test_standard_engine_runs_full_model_and_latent_kv(self):
        hardware = HardwareContext.load("data")
        doc = json.loads(Path("data/clusters/deepseek_v3/1a32f.json").read_text())
        doc["af"]["communication_domains"] = {
            "node_size": 8,
            "node_count": 5,
            "placement_strategy": "dedicated",
        }
        config = build_model_runtime(
            doc, hardware=hardware, model=hardware.model("deepseek-v3-671b-mla")
        )
        env = simpy.Environment()
        engine = LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=config.model,
            cluster_config=config.cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=16,
                batching="paged-attn",
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_num_seqs=8,
                latency_fallback="analytical_only",
            ),
        )
        request = _run_request(env, engine, decode_len=2)
        self.assertTrue(request.is_done)
        self.assertIsInstance(engine.af_executor, ModelAFExecutor)
        events = engine.af_executor.trace.events
        self.assertEqual(
            {e.layer_id for e in events if e.layer_id >= 0}, set(range(61))
        )
        self.assertEqual(
            {
                e.operation
                for e in events
                if e.layer_id == -1 and e.event_type == "compute"
            },
            {"embedding_lookup", "final_norm", "lm_head", "token_selection"},
        )
        self.assertTrue(
            all(layer.mla.cache_width == 576 for layer in config.plan.layers)
        )
        self.assertFalse(any(e.stage == "router" and e.layer_id < 3 for e in events))
        self.assertTrue(any(e.stage == "shared_expert" for e in events))

    def test_non_v3_mla_metadata_uses_the_same_execution_path(self):
        from dataclasses import replace

        config = configuration()
        model = replace(
            config.model,
            model_id="other-mla",
            num_layers=2,
            moe=replace(config.model.moe, first_k_dense_replace=1, num_moe_layers=1),
        )
        config = build_model_runtime(config.doc, hardware=config.hardware, model=model)
        executor, _, _, _ = simulate_decode(config, 1, 8, fallback="analytical_only")
        events = executor.trace.events
        self.assertFalse(any(e.layer_id == 0 and e.stage == "router" for e in events))
        self.assertTrue(any(e.layer_id == 1 and e.stage == "router" for e in events))
        attention = next(e for e in events if e.stage == "attention")
        self.assertGreater(len(attention.details["components"]), 1)
        self.assertAlmostEqual(
            attention.service_duration,
            sum(item["latency_us"] for item in attention.details["components"]) / 1e6,
        )
