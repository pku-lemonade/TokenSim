"""Global recording across execution modes without changing serving results."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import simpy

from TokenSim.cli import build_parser, main
from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.config import KVTransferConfig, ClusterConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.trace.config import TraceConfig
from tests.af.support.config import _with_prefill_pool, _cluster_dict
from tests.af.support.execution import af_test_hardware, _run_request
from tests.support.hardware import make_hardware, make_model
from TokenSim.llm.llm_request import Request


class ServingTraceTest(unittest.TestCase):

    def test_cached_layer_groups_expand_in_global_model_order(self):
        model = replace(make_model(), attention_windows=(4, 0, 4, 0))
        times = []
        for mode in ("off", "full"):
            hardware = make_hardware(models=[model])
            cluster = ClusterConfig(
                num_workers=1,
                networks={"local": "nvlink-test"},
                worker_groups=[
                    {
                        "role": "hybrid",
                        "hardware": "TestGPU",
                        "num_workers": 1,
                        "network": "local",
                    }
                ],
            )
            env = simpy.Environment()
            engine = LLMEngine(
                env,
                KVTransferConfig.default(),
                model,
                cluster,
                hardware,
                config=SimulationConfig(
                    trace={"mode": mode}, latency_fallback="analytical_only"
                ),
            )
            request = Request(1, 4, 1, 16)
            request.arrive(env)
            engine.add_requests_burst([request])
            env.run()
            engine.raise_if_failed()
            times.append(env.now)
            if mode == "full":
                order = list(
                    dict.fromkeys(
                        event.layer_id
                        for event in engine.trace_session.events()
                        if event.layer_id >= 0
                    )
                )
                self.assertEqual(order, [0, 1, 2, 3])
        self.assertEqual(times[0], times[1])

    def run_cli(self, directory, mode, *extra):
        args = build_parser().parse_args(
            [
                "--latency_backend",
                "analytical",
                "--verbose",
                "none",
                "--distribution",
                "burst",
                "--request_count",
                "2",
                "--prefill_mean_len",
                "16",
                "--decode_mean_len",
                "3",
                "--max_num_seqs",
                "2",
                "--results_path",
                str(directory),
                "--trace_mode",
                mode,
                *extra,
            ]
        )
        with redirect_stdout(io.StringIO()):
            engine = main(args)
        result = json.loads((Path(directory) / "result_inf.json").read_text())
        result.pop("simulator_wall_time")
        result.pop("trace", None)
        return engine, result

    def test_recording_modes_preserve_dense_and_moe_results(self):
        for extra in (
            (),
            (
                "--model",
                "data/models/moe-toy.yaml",
                "--cluster",
                "data/clusters/8_a100/moe_tp2_dp4.json",
            ),
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                results = []
                for mode in ("off", "validate", "full"):
                    path = Path(directory) / mode
                    with ExitStack() as checks:
                        if mode == "off":
                            for target in ("TraceSession", "TraceEvent"):
                                checks.enter_context(
                                    patch(
                                        "TokenSim.trace.session." + target,
                                        side_effect=AssertionError(
                                            "trace object constructed"
                                        ),
                                    )
                                )
                        engine, result = self.run_cli(path, mode, *extra)
                    results.append(result)
                    if mode == "off":
                        self.assertIsNone(engine.trace_session)
                    elif mode == "validate":
                        self.assertEqual(
                            engine.trace_session.recorder.events.retained_count, 0
                        )
                    else:
                        trace = json.loads((path / "result_inf.trace.json").read_text())
                        rows = [row for row in trace["traceEvents"] if row["ph"] == "X"]
                        self.assertTrue(rows)
                        self.assertTrue(
                            all(
                                row["args"]["dataset_version"] != "test-stub"
                                for row in rows
                            )
                        )
                        self.assertIn("attention", {row["name"] for row in rows})
                        self.assertTrue(all(row["dur"] >= 0 for row in rows))
                        self.assertFalse(trace["metadata"]["incomplete"])
                        recorded = engine.trace_session.recorder.events_for(0)
                        scope = [
                            e for e in recorded if e.worker_id == recorded[0].worker_id
                        ]
                        names = [e.operation for e in scope]
                        self.assertEqual(names[0], "embedding_lookup")
                        self.assertEqual(names.count("embedding_lookup"), 1)
                        self.assertEqual(names[-1], "token_selection")
                        layer = [e.operation for e in scope if e.layer_id == 0]
                        self.assertLess(layer.index("qkv"), layer.index("attention"))
                        self.assertLess(layer.index("attention"), layer.index("output"))
                        if extra:
                            self.assertEqual(names[1], "all_reduce")
                            self.assertEqual(
                                names[-3:], ["lm_head", "all_gather", "token_selection"]
                            )
                            self.assertEqual(names.count("all_gather"), 1)
                            self.assertEqual(
                                scope[1].details["collective_scope"], "tp_embedding"
                            )
                        else:
                            self.assertLess(
                                layer.index("ffn_norm"), layer.index("ffn_up")
                            )
                            self.assertLess(
                                layer.index("ffn_up"), layer.index("ffn_down")
                            )
                            norm = next(e for e in scope if e.operation == "final_norm")
                            self.assertEqual(norm.details["shape"]["num_tokens"], 32)
                self.assertEqual(results[0], results[1])
                self.assertEqual(results[0], results[2])

    def test_pd_transfer_has_producer_and_decode_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, result = self.run_cli(
                directory, "full", "--cluster", "data/clusters/8_a100/p2d5.json"
            )
            _, baseline = self.run_cli(
                Path(directory) / "off",
                "off",
                "--cluster",
                "data/clusters/8_a100/p2d5.json",
            )
            self.assertEqual(result, baseline)
            events = list(engine.trace_session.events())
            transfers = [event for event in events if event.operation == "kv_transfer"]
            self.assertTrue(transfers)
            by_id = {event.node_id: event for event in events}
            for transfer in transfers:
                self.assertTrue(transfer.predecessor_ids)
                self.assertTrue(
                    all(
                        by_id[key].end_at <= transfer.start_at + 1e-12
                        for key in transfer.predecessor_ids
                    )
                )

                self.assertTrue(
                    any(
                        transfer.node_id in event.predecessor_ids
                        for event in events
                        if event.phase == "decode"
                    )
                )

    def test_partial_prefill_normalizes_hidden_rows_without_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, _ = self.run_cli(
                directory,
                "full",
                "--request_count",
                "1",
                "--batching",
                "paged-attn",
                "--max_num_batched_tokens",
                "8",
                "--prefill_mean_len",
                "32",
                "--prefill_range_len",
                "0",
            )
            events = engine.trace_session.recorder.events_for(0)
            norm = next(event for event in events if event.operation == "final_norm")
            self.assertEqual(norm.details["shape"]["num_tokens"], 8)
            names = {event.operation for event in events}
            self.assertIn("embedding_lookup", names)
            self.assertFalse(names & {"lm_head", "all_gather", "token_selection"})

    def test_af_and_ordinary_prefill_share_one_export(self):
        with tempfile.TemporaryDirectory() as directory:
            times = []
            for mode in ("off", "validate", "full"):
                cluster = _with_prefill_pool(ClusterConfig(**_cluster_dict()))
                cluster.af.latency_provider = "operator_table"
                hardware = af_test_hardware(layers=1, capacity_gib=0.1)
                env = simpy.Environment()
                engine = LLMEngine(
                    env,
                    KVTransferConfig.default(),
                    hardware.model("TestModel"),
                    cluster,
                    hardware,
                    config=SimulationConfig(
                        trace=TraceConfig(mode=mode),
                        block_size=4,
                        latency_fallback="analytical_only",
                        batching="paged-attn",
                    ),
                )
                request = _run_request(env, engine)
                times.append(
                    (request.prefill_latency, request.decode_time_sum, env.now)
                )
                if mode == "full":
                    descriptor = engine.trace_session.export(directory)
                    rows = [
                        row
                        for row in json.loads(Path(descriptor["path"]).read_text())[
                            "traceEvents"
                        ]
                        if row["ph"] == "X"
                    ]
                    self.assertEqual(
                        {"prefill", "decode"}, {row["args"]["phase"] for row in rows}
                    )
                    self.assertTrue(
                        any(
                            row["args"].get("details", {}).get("parent_id")
                            for row in rows
                        )
                    )
                elif mode == "off":
                    self.assertFalse(engine.af_executor.trace.enabled)
            self.assertEqual(times[0], times[1])
            self.assertEqual(times[0], times[2])

    def test_bounded_run_does_not_export_predicted_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, _ = self.run_cli(directory, "full", "--sim_time", "0.000001")
            data = json.loads((Path(directory) / "result_inf.trace.json").read_text())
            self.assertTrue(data["metadata"]["incomplete"])
            partial = [row for row in data["traceEvents"] if row["ph"] == "X"]
            self.assertTrue(partial)
            self.assertTrue(
                all(row["args"]["details"]["incomplete"] for row in partial)
            )
            self.assertTrue(
                all(
                    row["ts"] + row["dur"] <= engine.env.now * 1e6 + 1e-8
                    for row in data["traceEvents"]
                    if row["ph"] == "X"
                )
            )
