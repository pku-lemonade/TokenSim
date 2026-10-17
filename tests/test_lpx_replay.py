"""Independent resource oracles and native controls for sampled DAG execution."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.concurrency import decode_probe
from scripts.lpx.hardware_overrides import reserve_memory
from scripts.lpx.native_config import configuration
from scripts.lpx.native_simulation import point
from scripts.lpx.replay_driver import ReplayConfig, ReplayDriver
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import cached_profile
from scripts.lpx.validation import validate_summary
from scripts.reproduce_lpx import main as reproduce_main


class ReplayKernelTest(unittest.TestCase):
    def setUp(self):
        self.kernel = ReplayKernel()
        self.addCleanup(self.kernel.close)

    def run_jobs(self, jobs):
        for nodes, arrival in jobs:
            self.kernel.submit(self.kernel.add(nodes), arrival)
        self.kernel.window(0)
        result = {}
        while self.kernel.active:
            row = self.kernel.advance()
            result[row.id] = (row.ready, row.start, row.end, row.released)
        return result

    def test_shared_compute_serializes_and_disjoint_compute_overlaps(self):
        profile = [
            ReplayNode((), ("gpu",), "gpu", 0, 1, 1),
            ReplayNode((0,), ("gpu",), "gpu", 0, 1, 1),
        ]
        rows = self.run_jobs([(profile, 0), (profile, 0)])
        self.assertEqual(sorted(row[2] for row in rows.values()), [3, 4])
        self.assertEqual(self.kernel.stats()["gpu"], (4, 3, 4))
        other = ReplayKernel()
        self.addCleanup(other.close)
        for resource in ("a", "b"):
            other.submit(other.add([ReplayNode((), (resource,), resource, 0, 2, 2)]), 0)
        self.assertEqual(other.advance().end, 2)
        self.assertEqual(other.advance().end, 2)

    def test_hop_completion_releases_injection_before_arrival(self):
        rows = self.run_jobs(
            [
                ([ReplayNode((), ("tx", "rx1"), "tx", 0, 11, 1)], 0),
                ([ReplayNode((), ("tx", "rx2"), "tx", 0, 11, 1)], 0),
            ]
        )
        self.assertEqual(rows[0], (0, 0, 11, 1))
        self.assertEqual(rows[1], (0, 1, 12, 2))
        self.assertEqual(self.kernel.stats()["tx"][0], 2)

    def test_atomic_fifo_blocks_bypass_without_blocking_disjoint_work(self):
        rows = self.run_jobs(
            [
                ([ReplayNode((), ("rx1",), "rx1", 0, 10, 10)], 0),
                ([ReplayNode((), ("tx", "rx1"), "tx", 0, 3, 3)], 1),
                ([ReplayNode((), ("tx", "rx2"), "tx", 0, 2, 2)], 2),
                ([ReplayNode((), ("other",), "other", 0, 2, 2)], 2),
            ]
        )
        self.assertEqual(rows[1][:3], (1, 10, 13))
        self.assertEqual(rows[2][:3], (2, 13, 15))
        self.assertEqual(rows[3][:3], (2, 2, 4))

    def test_external_timer_and_closing_window_are_not_skipped(self):
        window = dict(start=0)
        driver = ReplayDriver(ReplayConfig(Workload()), {}, window)
        self.addCleanup(driver.kernel.close)
        profile = driver.kernel.add([ReplayNode((), ("gpu",), "gpu", 0, 1, 1)])
        rows = []

        def stream(delay):
            yield driver.env.timeout(delay)
            row = yield driver.submit(profile)
            rows.append((row.ready, row.start, row.end))

        def close():
            yield driver.env.timeout(0.75)
            window["end"] = driver.env.now

        driver.env.process(stream(0))
        driver.env.process(stream(0.5))
        driver.env.process(close())
        driver.env.run()
        self.assertEqual(rows, [(0, 0, 1), (0.5, 1, 2)])
        self.assertEqual(driver.kernel.stats()["gpu"][:2], (0.75, 0.25))

    def test_invalid_and_cyclic_profiles_fail_without_native_memory_errors(self):
        with self.assertRaises(ConfigurationError):
            ReplayNode((), (), None, 0, float("nan"), 0)
        with self.assertRaises(ConfigurationError):
            self.kernel.add([ReplayNode((), ("x", "x"), "x", 0, 1, 1)])
        with self.assertRaisesRegex(SimulationStateError, "cyclic"):
            self.kernel.add(
                [
                    ReplayNode((1,), (), None, 0, 1, 0),
                    ReplayNode((0,), (), None, 0, 1, 0),
                ]
            )


class NativeReplayControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        base = cls.hardware.model("gpt-oss-2t-proxy")
        cls.model = replace(
            base,
            hidden_size=32,
            intermediate_size=64,
            num_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                base.moe,
                num_experts=4,
                num_experts_per_tok=4,
                num_moe_layers=4,
                moe_intermediate_size=64,
            ),
        )
        cls.workload = Workload(
            cached_tokens=16, new_tokens=2, output_tokens=8, prefill_chunk=2
        )
        cls.gpu = Layout(a_tp=2, a_pp=2, f_ep=2, f_pp=2)

    def test_four_complete_serving_protocols_preserve_accounting_and_native_rates(self):
        lpu = replace(
            self.gpu, a_device="groq_lp30", f_device="groq_lp30", a_pp=4, f_pp=4
        )
        cases = [
            Case("base", "rubin", self.gpu),
            Case(
                "af",
                "af",
                replace(self.gpu, f_device="groq_lp30", f_ep=4, colocated=False),
                self.gpu,
            ),
            Case("pd", "pd", lpu, self.gpu),
            Case("draft", "draft", self.gpu, draft=lpu),
        ]
        with tempfile.TemporaryDirectory() as cache:
            for case in cases:
                with self.subTest(mode=case.mode):
                    settings = dict(
                        micro_batch=1, warmup_steps=1, measurement_seconds=1e-9
                    )
                    native = point(
                        self.hardware, self.model, self.workload, case, 2, **settings
                    )
                    replay = point(
                        self.hardware,
                        self.model,
                        self.workload,
                        case,
                        2,
                        execution_factory=ReplayConfig(
                            self.workload, samples=1, cache=cache
                        ),
                        **settings,
                    )
                    for key in ("inventory", "power_watts"):
                        self.assertEqual(replay[key], native[key])
                    for key in ("decode_tokens", "output_tokens"):
                        self.assertEqual(
                            replay["steady_state"][key], native["steady_state"][key]
                        )
                    for key in ("output_tokens_per_s", "tokens_per_s_per_user"):
                        self.assertLess(abs(replay[key] / native[key] - 1), 0.01)
                    self.assertEqual(replay["native_executor"], "SampledNativeReplay")

    def test_cached_native_profile_reconstructs_and_detects_body_corruption(self):
        runtime = configuration(
            self.hardware, self.model, self.gpu, namespace="profile"
        )
        settings = dict(
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=20,
            dp_rank=0,
            seed=42,
            sample=0,
        )
        with tempfile.TemporaryDirectory() as cache:
            original = cached_profile(runtime, "profile", cache, **settings)
            restored = cached_profile(runtime, "profile", cache, **settings)
            self.assertEqual(original, restored)
            path = next(Path(cache).glob("*.json.gz"))
            with gzip.open(path, "rt") as stream:
                value = json.load(stream)
            value["nodes"][0][4] += 1
            with gzip.open(path, "wt") as stream:
                json.dump(value, stream)
            with self.assertRaisesRegex(ConfigurationError, "body fingerprint"):
                cached_profile(runtime, "profile", cache, **settings)

    def test_effective_memory_overrides_change_cache_keys_and_preserve_catalog(self):
        with tempfile.TemporaryDirectory() as cache:
            profiles = []
            for reserve in (0, 64_000_000):
                hardware = reserve_memory(self.hardware, {"rubin": reserve})
                runtime = configuration(
                    hardware, self.model, self.gpu, namespace="reserve"
                )
                profiles.append(
                    cached_profile(
                        runtime,
                        "reserve",
                        cache,
                        batch=1,
                        queries=1,
                        phase="decode",
                        emit_logits=True,
                        context=20,
                        dp_rank=0,
                        seed=42,
                        sample=0,
                    )
                )
                self.assertEqual(
                    hardware.device("rubin").usable_memory_bytes,
                    self.hardware.device("rubin").usable_memory_bytes - reserve,
                )
            self.assertNotEqual(
                profiles[0].metadata["input_sha256"],
                profiles[1].metadata["input_sha256"],
            )
            self.assertEqual(profiles[0].nodes, profiles[1].nodes)
            with self.assertRaises(ConfigurationError):
                reserve_memory(self.hardware, {"rubin": -1})

    def test_decode_probe_uses_same_shared_resources_with_partial_cohort(self):
        runtime = configuration(
            self.hardware, self.model, self.gpu, namespace="probe", micro_batch=2
        )
        with tempfile.TemporaryDirectory() as cache:
            settings = dict(micro_batch=2, context=16, steps=4, warmup_steps=1)
            native = decode_probe(runtime, 5, **settings)
            replay = decode_probe(
                runtime,
                5,
                **settings,
                execution_factory=ReplayConfig(self.workload, samples=1, cache=cache),
            )
            for key in ("decode_tokens_per_s", "tokens_per_s_per_user"):
                self.assertLess(abs(replay[key] / native[key] - 1), 0.01)
            self.assertEqual(replay["memory"], native["memory"])
            self.assertGreater(replay["mean_compute_utilization"], 0)

    def test_memory_allowance_preserves_registered_operator_data_without_aliasing_index(
        self,
    ):
        hardware = replace(self.hardware, _packages={}, _package_index={})
        package = SimpleNamespace(device_id="rubin", backend="fixture")
        hardware.register_package(package)
        adjusted = reserve_memory(hardware, {"rubin": 64_000_000})
        self.assertIs(adjusted.operator_package("rubin", "fixture"), package)
        adjusted.register_package(SimpleNamespace(device_id="rubin", backend="extra"))
        self.assertEqual(hardware.available_backends("rubin"), ["fixture"])
        self.assertEqual(adjusted.available_backends("rubin"), ["fixture", "extra"])

    def test_cli_and_audit_distinguish_sampled_execution_and_check_provenance(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "scripts.reproduce_lpx.HardwareContext.load", return_value=self.hardware
            ),
            patch.object(self.hardware, "model", return_value=self.model),
            redirect_stdout(StringIO()),
        ):
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        workload=asdict(self.workload),
                        batches=[1, 2],
                        cases=[asdict(Case("base", "rubin", self.gpu))],
                    )
                )
            )
            self.assertEqual(
                reproduce_main(
                    [
                        "--config",
                        str(config),
                        "--output",
                        str(root / "run"),
                        "--skip-audit",
                        "--execution-backend",
                        "sampled-replay",
                        "--replay-cache",
                        str(root / "cache"),
                        "--replay-samples",
                        "1",
                        "--measurement-seconds",
                        "0.000000001",
                        "--lp30-reserve-bytes",
                        "64000000",
                    ]
                ),
                0,
            )
            payload = json.loads((root / "run/summary.json").read_text())
            audit = validate_summary(payload)
            self.assertEqual(audit["status"], "accounting_passed")
            self.assertEqual(
                audit["numerical_accuracy"], "requires_separate_native_calibration"
            )
            self.assertEqual(
                payload["hardware"]["rubin"], self.hardware.device("rubin").to_dict()
            )
            for mutation in ("executor", "fingerprint", "missing_events"):
                changed = deepcopy(payload)
                if mutation == "executor":
                    changed["executor"] = "production_ModelAFExecutor"
                elif mutation == "fingerprint":
                    changed["results"][0]["replay_profiles"]["profiles"][0][
                        "input_sha256"
                    ] = "bad"
                else:
                    del changed["results"][0]["token_events"]
                with (
                    self.subTest(mutation=mutation),
                    self.assertRaises(ConfigurationError),
                ):
                    validate_summary(changed)
