from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import simpy

from TokenSim.config.psla_config import PSLAConfig
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.workload.loaders import load_workload
from util.request import LLMSource


class _RecordingEngine:
    def __init__(self) -> None:
        self.arrivals: list[tuple[float, int]] = []

    def add_requests(self, requests: list[Request]) -> None:
        for request in requests:
            self.arrivals.append((request.arrival_time, request.id))

    def add_requests_burst(self, requests: list[Request]) -> None:
        self.add_requests(requests)


class TraceTimestampScalingTest(unittest.TestCase):
    def _model_config(self) -> PSLAConfig:
        return PSLAConfig.from_file("./data/psla/llama-7b.json")

    def _args(
        self,
        *,
        workload_type: str,
        dataset_path: str | None = None,
        request_count: int | None = None,
        trace_timestamp_scale: float | None = None,
        trace_target_qps: float | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            workload_type=workload_type,
            dataset_path=dataset_path,
            request_count=request_count,
            dataset_skip_count=0,
            random_seed=0,
            trace_timestamp_scale=trace_timestamp_scale,
            trace_target_qps=trace_target_qps,
        )

    def _write_json_pairs(self, rows: list) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "trace.json"
        path.write_text(json.dumps(rows))
        return str(path)

    def _write_qwen_jsonl(self, rows: list[dict]) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "trace.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        return str(path)

    def test_direct_scale_multiplies_zero_origin_timestamps(self):
        path = self._write_json_pairs(
            [
                [16, 1, {"arrival_time": 0.0}],
                [16, 1, {"arrival_time": 1.5}],
                [16, 1, {"arrival_time": 3.0}],
            ]
        )
        workload = load_workload(
            self._args(
                workload_type="json_pairs",
                dataset_path=path,
                request_count=3,
                trace_timestamp_scale=2.0,
            ),
            self._model_config(),
        )

        self.assertEqual([request.arrival_time for request in workload], [0.0, 3.0, 6.0])

    def test_target_qps_uses_interval_count_over_last_timestamp(self):
        path = self._write_json_pairs(
            [
                [16, 1, {"arrival_time": 0.0}],
                [16, 1, {"arrival_time": 0.02}],
                [16, 1, {"arrival_time": 0.04}],
            ]
        )
        workload = load_workload(
            self._args(
                workload_type="json_pairs",
                dataset_path=path,
                request_count=3,
                trace_target_qps=10.0,
            ),
            self._model_config(),
        )

        self.assertEqual([request.arrival_time for request in workload], [0.0, 0.1, 0.2])

    def test_invalid_scale_and_target_qps_values_raise(self):
        path = self._write_json_pairs([[16, 1, {"arrival_time": 0.0}]])

        invalid_cases = [
            {"trace_timestamp_scale": 0.0},
            {"trace_timestamp_scale": math.inf},
            {"trace_timestamp_scale": math.nan},
            {"trace_target_qps": 0.0},
            {"trace_target_qps": math.inf},
            {"trace_target_qps": math.nan},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(WorkloadValidationError):
                    load_workload(
                        self._args(
                            workload_type="json_pairs",
                            dataset_path=path,
                            request_count=1,
                            **kwargs,
                        ),
                        self._model_config(),
                    )

    def test_mutually_exclusive_scale_arguments_raise(self):
        path = self._write_json_pairs([[16, 1, {"arrival_time": 0.0}]])

        with self.assertRaises(WorkloadValidationError):
            load_workload(
                self._args(
                    workload_type="json_pairs",
                    dataset_path=path,
                    request_count=1,
                    trace_timestamp_scale=2.0,
                    trace_target_qps=1.0,
                ),
                self._model_config(),
            )

    def test_scaling_requires_timestamped_dataset_records(self):
        path = self._write_json_pairs([[16, 1], [16, 1]])

        with self.assertRaises(WorkloadValidationError):
            load_workload(
                self._args(
                    workload_type="json_pairs",
                    dataset_path=path,
                    request_count=2,
                    trace_timestamp_scale=2.0,
                ),
                self._model_config(),
            )

    def test_synthetic_workload_rejects_trace_scaling(self):
        with self.assertRaises(WorkloadValidationError):
            load_workload(
                self._args(
                    workload_type="synthetic",
                    request_count=2,
                    trace_timestamp_scale=2.0,
                ),
                self._model_config(),
            )

    def test_target_qps_requires_inferable_original_qps(self):
        single_timestamp_path = self._write_json_pairs([[16, 1, {"arrival_time": 0.0}]])
        zero_span_path = self._write_json_pairs(
            [
                [16, 1, {"arrival_time": 0.0}],
                [16, 1, {"arrival_time": 0.0}],
            ]
        )

        with self.assertRaises(WorkloadValidationError):
            load_workload(
                self._args(
                    workload_type="json_pairs",
                    dataset_path=single_timestamp_path,
                    request_count=1,
                    trace_target_qps=1.0,
                ),
                self._model_config(),
            )
        with self.assertRaises(WorkloadValidationError):
            load_workload(
                self._args(
                    workload_type="json_pairs",
                    dataset_path=zero_span_path,
                    request_count=2,
                    trace_target_qps=1.0,
                ),
                self._model_config(),
            )

    def test_qwen_jsonl_normalizes_then_scales_timestamps(self):
        path = self._write_qwen_jsonl(
            [
                {"timestamp": 10.0, "input_length": 16, "output_length": 1},
                {"timestamp": 11.5, "input_length": 16, "output_length": 1},
                {"timestamp": 12.0, "input_length": 16, "output_length": 1},
            ]
        )
        workload = load_workload(
            self._args(
                workload_type="qwen_jsonl",
                dataset_path=path,
                request_count=3,
                trace_timestamp_scale=2.0,
            ),
            self._model_config(),
        )

        self.assertEqual([request.arrival_time for request in workload], [0.0, 3.0, 4.0])

    def test_inter_arrival_values_are_not_scaled_in_mixed_workload(self):
        path = self._write_json_pairs(
            [
                [16, 1, {"arrival_time": 1.0}],
                [16, 1, {"inter_arrival_time": 2.0}],
            ]
        )
        workload = load_workload(
            self._args(
                workload_type="json_pairs",
                dataset_path=path,
                request_count=2,
                trace_timestamp_scale=3.0,
            ),
            self._model_config(),
        )

        self.assertEqual(workload[0].arrival_time, 3.0)
        self.assertIsNone(workload[1].arrival_time)
        self.assertEqual(workload[1].inter_arrival_time, 2.0)

    def test_llm_source_replays_scaled_timestamps_in_order(self):
        path = self._write_json_pairs(
            [
                [16, 1, {"arrival_time": 2.0}],
                [16, 1, {"arrival_time": 0.0}],
                [16, 1, {"arrival_time": 5.0}],
            ]
        )
        workload = load_workload(
            self._args(
                workload_type="json_pairs",
                dataset_path=path,
                request_count=3,
                trace_timestamp_scale=2.0,
            ),
            self._model_config(),
        )
        requests = workload.to_requests(block_size=16)
        env = simpy.Environment()
        engine = _RecordingEngine()

        env.process(LLMSource(env, engine, requests, qps=1000, distribution="uniform"))
        env.run()

        self.assertEqual(engine.arrivals, [(0.0, 1), (4.0, 0), (10.0, 2)])


if __name__ == "__main__":
    unittest.main()
