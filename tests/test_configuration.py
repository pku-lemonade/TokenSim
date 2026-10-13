"""Configuration boundaries reject invalid inputs through project errors."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from TokenSim.cli import cli
from TokenSim.config.router_config import load_router_mapping
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import LookupPolicy
from TokenSim.operator_data.schema import AxisSpec
from TokenSim.placement.consistent_hash import FixedWorkerRouter


class RoutingConfigurationTest(unittest.TestCase):
    def test_mapping_file_accepts_direct_and_wrapped_integer_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            for data in ({"chat": "0"}, {"mapping": {"chat": 0}}):
                path.write_text(json.dumps(data))
                self.assertEqual(load_router_mapping(path), {"chat": 0})

    def test_cli_reports_invalid_mapping_files_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            for contents in (None, "{", "[]", "null", '{"mapping": []}'):
                with self.subTest(contents=contents):
                    if contents is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(contents)
                    output = io.StringIO()
                    with redirect_stderr(output):
                        code = cli(
                            [
                                "--latency_backend",
                                "analytical",
                                "--router_mapping_path",
                                str(path),
                                "--verbose",
                                "none",
                            ]
                        )
                    self.assertEqual(code, 1)
                    self.assertIn("router mapping", output.getvalue())
                    self.assertNotIn("Traceback", output.getvalue())

    def test_rank_validation_applies_to_configuration_and_router(self):
        workers = [SimpleNamespace(dp_rank=0)]
        for value in (0.9, 1.0, True, None, -1, "bad", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    SimulationConfig(router_mapping={"chat": value})
                with self.assertRaises(ConfigurationError):
                    FixedWorkerRouter(workers, {"chat": value})
        with self.assertRaisesRegex(ConfigurationError, "unknown DP rank"):
            FixedWorkerRouter(workers, {"chat": 1})
        for mapping in ([], "bad"):
            with self.assertRaises(ConfigurationError):
                SimulationConfig(router_mapping=mapping)

    def test_valid_router_mapping_selects_the_expected_worker(self):
        workers = [SimpleNamespace(dp_rank=rank, id=rank) for rank in (0, 1)]
        router = FixedWorkerRouter(workers, {"chat": "1"})
        request = SimpleNamespace(source=None, chat_id="chat")
        self.assertIs(router.select(request), workers[1])
        self.assertEqual(request.dp_rank, 1)


class LookupConfigurationTest(unittest.TestCase):
    def test_policy_and_axis_reject_invalid_extrapolation_bounds(self):
        for value in (0, -1, 0.9, True, "16", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    LookupPolicy(max_extrapolation_ratio=value)
                with self.assertRaises(ConfigurationError):
                    AxisSpec("m", max_extrapolation_ratio=value)

    def test_valid_bounds_include_one_and_axis_default(self):
        self.assertEqual(
            LookupPolicy(max_extrapolation_ratio=1).max_extrapolation_ratio, 1
        )
        self.assertEqual(
            AxisSpec("m", max_extrapolation_ratio=1).max_extrapolation_ratio, 1
        )
        self.assertIsNone(AxisSpec("m").max_extrapolation_ratio)

    def test_invalid_policy_choices_use_configuration_errors(self):
        with self.assertRaises(ConfigurationError):
            LookupPolicy(extrapolate="unknown")
        with self.assertRaises(ConfigurationError):
            LookupPolicy(interpolate="false")
        with self.assertRaises(ConfigurationError):
            AxisSpec("m", scale="unknown")


class MTPConfigurationTest(unittest.TestCase):
    def test_cli_rejects_insufficient_mtp_budget_before_starting_simulation(self):
        for drafts, acceptance in ((1, 2), (3, 1)):
            with self.subTest(drafts=drafts, acceptance=acceptance):
                output = io.StringIO()
                with redirect_stderr(output):
                    code = cli(
                        [
                            "--batching",
                            "paged-attn",
                            "--mtp_num_speculative_tokens",
                            str(drafts),
                            "--mtp_acceptance_length",
                            str(acceptance),
                            "--max_num_batched_tokens",
                            str(drafts),
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertIn("max_num_batched_tokens", output.getvalue())
                self.assertIn("MTP", output.getvalue())
                self.assertNotIn("Traceback", output.getvalue())
