"""Scheduling order, independent timing, deadlines and durable result semantics."""

from contextlib import redirect_stdout
import heapq
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import simpy

from benchmark import build_parser, main
from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.llm_engine import Message, Task, Worker, LLMWorker
from TokenSim.llm.llm_scheduler import LLMDynamicScheduler, LLMStaticScheduler
from TokenSim.llm.llm_request import Request
from TokenSim.workload.loaders import load_qwen_jsonl_workload

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    def test_mooncake_cache_is_shared_within_engine_and_isolated_between_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cluster = path / "cluster.json"
            cluster.write_text(
                json.dumps(
                    {
                        "num_workers": 1,
                        "networks": {"node": "ethernet100Gb"},
                        "worker_groups": [
                            {
                                "role": "hybrid",
                                "hardware": "a100_sxm_80g",
                                "num_workers": 1,
                                "network": "node",
                            }
                        ],
                        "kv_cache_capacity_tokens_total": 80,
                    }
                )
            )
            connector = path / "kv.json"
            connector.write_text(
                json.dumps(
                    {
                        "kv_connector": "MooncakeStoreConnector",
                        "engine_id": "reused-id",
                        "kv_connector_extra_config": {"memory_capacity_blocks": 128},
                    }
                )
            )
            dataset = path / "requests.json"
            dataset.write_text(
                json.dumps(
                    [
                        [64, 2, {"hash_ids": [1, 2, 3, 4]}],
                        [64, 2, {"hash_ids": [5, 6, 7, 8]}],
                        [64, 2, {"hash_ids": [1, 2, 3, 4]}],
                    ]
                )
            )
            services = []
            for run in range(2):
                target = path / str(run)
                args = self.args(
                    target,
                    "--distribution",
                    "uniform",
                    "--qps",
                    "1",
                    "--batching",
                    "paged-attn",
                    "--request_count",
                    "3",
                    "--max_num_batched_tokens",
                    "32",
                    "--cluster",
                    str(cluster),
                    "--kv_transfer_config",
                    str(connector),
                    "--dataset_path",
                    str(dataset),
                    "--workload_type",
                    "json_pairs",
                )
                engine = self.run_main(args)
                result = json.loads((target / "result_1.0.json").read_text())
                self.assertEqual(result["request_count"], 3)
                store = engine.workers[0].connector
                self.assertGreater(store.mooncake_stats.mooncake_memory_hit_tokens, 0)
                self.assertEqual(store.mooncake_stats.store_hit_count, 1)
                services.append(store.service)
            self.assertIsNot(services[0], services[1])

    def test_model_geometry_and_routing_are_independent_of_workload_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            workload = path / "workload.json"
            workload.write_text(
                json.dumps(
                    {
                        "qps": 9,
                        "prefill_mean_len": 16,
                        "decode_mean_len": 2,
                        "first_token_latency": {"max": 7},
                    }
                )
            )
            args = build_parser().parse_args(
                [
                    "--model",
                    str(ROOT / "data/models/moe-toy.yaml"),
                    "--cluster",
                    str(ROOT / "data/clusters/8_a100/moe_tp2_dp4.json"),
                    "--workload_config",
                    str(workload),
                    "--request_count",
                    "2",
                    "--distribution",
                    "burst",
                    "--latency_backend",
                    "analytical",
                    "--moe_routing_distribution",
                    "hot",
                    "--moe_hot_experts",
                    "0,1",
                    "--moe_hot_expert_fraction",
                    "0.8",
                    "--verbose",
                    "none",
                    "--results_path",
                    str(path),
                ]
            )
            engine = self.run_main(args)
            self.assertEqual(engine.model_spec.hidden_size, 128)
            self.assertEqual(engine.model_spec.moe.num_experts, 4)
            self.assertEqual(engine.moe_config.routing.hot_experts, [0, 1])
            self.assertEqual(engine.workload_config.qps, 9)
            self.assertEqual(engine.workload_config.first_token_latency.max, 7)
            result = json.loads((path / "result_inf.json").read_text())
            self.assertEqual(result["prefill_lens"], [16, 16])
            self.assertEqual(result["decode_lens"], [2, 2])
            workload.write_text(json.dumps({"hidden_size": 256}))
            with self.assertRaisesRegex(ConfigurationError, "hidden_size"):
                WorkloadConfig.from_file(workload)

    def args(self, directory, *extra):
        return build_parser().parse_args(
            [
                "--latency_backend",
                "analytical",
                "--verbose",
                "none",
                "--distribution",
                "burst",
                "--request_count",
                "8",
                "--prefill_mean_len",
                "16",
                "--decode_mean_len",
                "3",
                "--max_num_seqs",
                "2",
                "--results_path",
                str(directory),
                *extra,
            ]
        )

    def run_main(self, args):
        with redirect_stdout(io.StringIO()):
            return main(args)

    def test_equal_priority_messages_are_fifo_and_cancel_has_priority(self):
        messages = [Message(Task.ADD, sender_id=i) for i in range(40)]
        heapq.heapify(messages)
        messages.append(Message(Task.CANCEL, sender_id=-1))
        heapq.heapify(messages)
        self.assertEqual(heapq.heappop(messages).task, Task.CANCEL)
        self.assertEqual(
            [heapq.heappop(messages).sender_id for _ in range(40)], list(range(40))
        )

    def test_batch_modes_respect_sequence_limits_and_waiting_order(self):
        for cls in (LLMDynamicScheduler, LLMStaticScheduler):
            scheduler = cls(max_num_seqs=2)
            requests = [Request(i, 16 + i, 3, 16) for i in range(5)]
            scheduler.add_requests(requests)
            running, _ = scheduler.schedule()
            self.assertEqual([r.id for r in running], [0, 1])
            self.assertEqual([r.id for r in scheduler.waiting], [2, 3, 4])
            scheduler.running.clear()
            self.assertEqual([r.id for r in scheduler.schedule()[0]], [2, 3])
        with self.assertRaises(ConfigurationError):
            SimulationConfig(max_num_seqs=0)

    def test_main_batch_modes_count_output_tokens_and_keep_timing_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            engines = []
            for batching in ("dynamic", "static", "paged-attn"):
                path = Path(directory) / batching
                args = self.args(path, "--batching", batching)
                engine = self.run_main(args)
                result = json.loads((path / "result_inf.json").read_text())
                self.assertEqual(result["request_count"], 8)
                self.assertEqual(result["notdone"], [])
                self.assertEqual(len(engine.timing_recorder.timing.time), 8)
                self.assertAlmostEqual(
                    result["output_token_ps"] * result["duration"], 24
                )
                engines.append(engine)
            self.assertIsNot(
                engines[0].timing_recorder.timing, engines[1].timing_recorder.timing
            )
            self.assertEqual(len(engines[0].timing_recorder.timing.time), 8)

    def test_pd_dynamic_batching_charges_every_transfer_plan(self):
        issued, charged = [], []
        original_send, original_prepare = Worker.send_task, LLMWorker.prepare_step

        def send(worker, target, task, requests=None, metadata=None):
            if task == Task.ADD and metadata is not None:
                issued.extend(plan.latency for plan in metadata.loads)
            return original_send(worker, target, task, requests, metadata)

        def prepare(worker):
            running, latency = original_prepare(worker)
            charged.append(latency)
            return running, latency

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(
                directory,
                "--request_count",
                "48",
                "--max_num_seqs",
                "4",
                "--cluster",
                str(ROOT / "data/clusters/8_a100/p2d5.json"),
            )
            with (
                patch.object(Worker, "send_task", send),
                patch.object(LLMWorker, "prepare_step", prepare),
            ):
                engine = self.run_main(args)
            self.assertGreater(sum(issued), 0)
            self.assertAlmostEqual(sum(issued), sum(charged))
            self.assertEqual(len(engine.timing_recorder.timing.time), 48)

    def test_cutoff_and_failure_always_preserve_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = self.args(path, "--sim_time", "0.000001")
            self.run_main(args)
            result = json.loads((path / "result_inf.json").read_text())
            self.assertEqual(result["termination_reason"], "sim_time_limit")
            self.assertTrue(result["notdone"])
            self.assertEqual(result["simulated_time"], 0.000001)
            args = self.args(path)
            with patch(
                "TokenSim.llm.llm_engine.LLMEngine.raise_if_failed",
                side_effect=SimulationStateError("conservation failed"),
            ):
                with self.assertRaises(SimulationStateError):
                    self.run_main(args)
            result = json.loads((path / "result_inf.json").read_text())
            self.assertEqual(result["termination_reason"], "failed")
            self.assertIn("conservation failed", result["failure"])
            self.assertTrue((path / "failure.json").is_file())
            self.assertTrue((path / "failed.txt").is_file())

    def test_agentx_uses_regular_model_and_warmup_cutoff_does_not_drain_past_limit(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(
                directory,
                "--batching",
                "paged-attn",
                "--workload_type",
                "agentx_weka",
                "--dataset_path",
                str(ROOT / "tests/fixtures/agentx_weka.jsonl"),
                "--block_size",
                "64",
                "--agentx_profile_duration",
                "2",
                "--sim_time",
                "0.001",
            )
            self.run_main(args)
            result = json.loads((Path(directory) / "agentx_c1.json").read_text())
            self.assertEqual(result["termination_reason"], "sim_time_limit")
            self.assertEqual(result["agentx_metrics"]["random_seed"], args.random_seed)
            windows = [
                json.loads(line)
                for line in (Path(directory) / "replay/cache_windows.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertLessEqual(windows[-1]["simulation_time"], 0.001)

    def test_duplicate_trace_ids_cannot_collide_with_fallback_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"request_id": i, "input_length": 8, "output_length": 2})
                    for i in (1, 1, 2, 0, 0)
                )
            )
            with redirect_stdout(io.StringIO()):
                requests = load_qwen_jsonl_workload(str(path), 5)
            ids = [r.request_id for r in requests]
            self.assertEqual(len(set(ids)), 5)
