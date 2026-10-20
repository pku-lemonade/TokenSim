"""Native result audits reject unit, capacity, transfer and frontier corruption."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
import csv
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.native_simulation import point
from scripts.lpx.frontier import pareto
from scripts.lpx.validation import power_snapshot, validate_summary
from scripts.check_lpx_results import main as audit_main
from scripts.reproduce_lpx import main as reproduce_main


class LPXResultAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        base = cls.hardware.model("gpt-oss-2t-expert-proxy")
        model = replace(
            base,
            hidden_size=32,
            intermediate_size=64,
            num_layers=4,
            attention_windows=(),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                base.moe, num_experts=8, num_moe_layers=4, moe_intermediate_size=64
            ),
        )
        cls.model = model
        workload = Workload(
            cached_tokens=16, new_tokens=2, output_tokens=8, prefill_chunk=2
        )
        layout = Layout(a_tp=2, f_ep=2)
        cases = [
            Case("base", "rubin", layout),
            Case(
                "af",
                "af",
                replace(layout, f_device="groq_lp30", f_ep=4, colocated=False),
                layout,
            ),
            Case(
                "draft",
                "draft",
                layout,
                draft=replace(
                    layout,
                    a_device="groq_lp30",
                    f_device="groq_lp30",
                ),
            ),
        ]
        rows = [
            point(
                cls.hardware,
                model,
                workload,
                case,
                1,
                warmup_steps=1,
                measurement_seconds=1e-9,
                draft_spec=model,
            )
            for case in cases
        ]
        baseline = rows[0]["tokens_per_s_per_mw"]
        for row in rows:
            row["normalized_throughput"] = row["tokens_per_s_per_mw"] / baseline
        cls.payload = dict(
            schema="tokensim.lpx_projection",
            version=2,
            executor="production_ModelAFExecutor",
            workload=asdict(workload),
            model=model.to_dict(),
            cases=[asdict(c) for c in cases],
            hardware={
                name: cls.hardware.device(name).to_dict()
                for name in ("rubin", "groq_lp30")
            },
            power_overrides={},
            power_model=power_snapshot(cls.hardware, {}),
            facility_watts=0,
            run_settings={
                "a_replicas": 1,
                "client_counts": [1],
                "event_accounting_required": True,
            },
            normalization_tokens_per_s_per_mw=baseline,
            results=rows,
            frontiers={
                mode: pareto([row for row in rows if row["mode"] == mode])
                for mode in ("rubin", "draft", "af", "pd")
            },
        )

    def test_real_native_points_and_explicit_older_catalog_audit(self):
        report = validate_summary(json.loads(json.dumps(self.payload)))
        self.assertEqual(report["feasible"], 3)
        self.assertEqual(report["grid_coverage"], "declared")
        older = deepcopy(self.payload)
        del older["power_model"]
        with self.assertRaisesRegex(ConfigurationError, "power snapshot"):
            validate_summary(older)
        self.assertEqual(validate_summary(older, hardware=self.hardware), report)

    def test_independent_result_invariants_reject_corruption(self):
        def remove_compute_slot(payload):
            payload["results"][0]["services"]["target"]["resource_capacities"].pop(
                "worker:1"
            )
            payload["results"][0]["steady_state"]["resource_utilization"].pop(
                "target/worker:1", None
            )

        changes = [
            lambda p: p["results"][0]["inventory"].update(rubin=999),
            lambda p: p["results"][0].update(power_watts=1),
            lambda p: p["results"][0].update(tokens_per_s_per_user=1e30),
            remove_compute_slot,
            lambda p: p["results"][0]["steady_state"].update(output_tokens=999),
            lambda p: p["results"][0]["steady_state"]["resource_utilization"].update(
                {"target/worker:0": 1.1}
            ),
            lambda p: p["results"][0].update(turns=[]),
            lambda p: p["results"][1]["handoff_events"][0].update(bytes=1),
            lambda p: p["results"][0].update(normalized_throughput=123),
            lambda p: p["frontiers"].update(rubin=[]),
            lambda p: p["cases"].append(p["cases"][0]),
            lambda p: p["run_settings"].update(client_counts=[1, 2]),
            lambda p: p["results"][2]["verification_events"].pop(),
            lambda p: p["results"][0]["token_events"][0].update(tokens=999),
            lambda p: p["results"][0]["decode_intervals"][0].update(tokens=999),
            lambda p: p["results"][0].pop("token_events"),
            lambda p: p["results"][0].pop("decode_intervals"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(corruption=index):
                broken = deepcopy(self.payload)
                change(broken)
                with self.assertRaises(ConfigurationError):
                    validate_summary(broken)

    def test_original_workload_check_rejects_smoke_substitution(self):
        with self.assertRaisesRegex(ConfigurationError, "original figure workload"):
            validate_summary(self.payload, original_workload=True)

    def test_original_workload_cannot_substitute_a_fixed_decode_window(self):
        payload = deepcopy(self.payload)
        payload["workload"].update(
            cached_tokens=400000, new_tokens=4000, output_tokens=400
        )
        payload["run_settings"]["measurement_turns"] = 0
        with self.assertRaisesRegex(ConfigurationError, "complete measured turns"):
            validate_summary(payload, original_workload=True)

    def test_older_accounting_has_an_explicitly_limited_audit_scope(self):
        older = deepcopy(self.payload)
        older["run_settings"].pop("event_accounting_required")
        for row in older["results"]:
            del row["token_events"], row["decode_intervals"]
        self.assertEqual(
            validate_summary(older)["event_accounting"], "legacy_turn_totals"
        )
        self.assertEqual(
            validate_summary(self.payload)["event_accounting"],
            "token_events_and_intervals",
        )

    def test_external_grid_checks_older_results_and_declared_layouts(self):
        older = deepcopy(self.payload)
        del older["run_settings"]["client_counts"]
        self.assertEqual(validate_summary(older)["grid_coverage"], "not_recorded")
        self.assertEqual(
            validate_summary(older, expected_batches=[1])["grid_coverage"], "external"
        )
        for batches in ([1, 2], [1, 1], [0], []):
            with self.subTest(batches=batches), self.assertRaises(ConfigurationError):
                validate_summary(older, expected_batches=batches)
        cases = [Case("base", "rubin", Layout(a_tp=1, f_ep=1))]
        with self.assertRaisesRegex(ConfigurationError, "saved cases differ"):
            validate_summary(older, expected_cases=cases)
        older["results"].pop()
        older["frontiers"]["draft"] = []
        with self.assertRaisesRegex(ConfigurationError, "missing declared case"):
            validate_summary(older)

    def test_cli_exports_audited_raw_throughput_and_hashed_evidence(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            output = Path(directory)
            summary = output / "summary.json"
            source = json.dumps(self.payload).encode()
            summary.write_bytes(source)
            config = output / "config.json"
            config.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        workload=self.payload["workload"],
                        batches=[1],
                        cases=self.payload["cases"],
                    )
                )
            )
            args = ["--summary", str(summary), "--config", str(config)]
            self.assertEqual(audit_main([*args, "--export", str(output)]), 0)
            evidence = json.loads((output / "evidence.json").read_text())
            self.assertEqual(
                evidence["source_summary"]["sha256"], hashlib.sha256(source).hexdigest()
            )
            self.assertEqual(evidence["audit"]["grid_coverage"], "external")
            self.assertEqual(summary.read_bytes(), source)
            measured = evidence["results"][2]["measured_verification"]
            self.assertEqual(
                measured["committed_tokens"],
                self.payload["results"][2]["steady_state"]["decode_tokens"],
            )
            with (output / "points.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                float(rows[0]["output_tokens_per_s"]),
                self.payload["results"][0]["output_tokens_per_s"],
            )
            for extra in (
                ["--batches", "1", "2"],
                ["--cases", "missing"],
                ["--output", str(summary)],
                ["--export", str(output), "--output", str(output / "evidence.json")],
            ):
                with self.subTest(extra=extra):
                    self.assertEqual(audit_main([*args, *extra]), 1)
                    self.assertEqual(summary.read_bytes(), source)

    def test_native_cli_records_and_checks_requested_client_grid(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "scripts.reproduce_lpx.HardwareContext.load", return_value=self.hardware
            ),
            patch.object(self.hardware.models, "resolve", return_value=self.model),
            redirect_stdout(StringIO()),
        ):
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        workload=self.payload["workload"],
                        batches=[1],
                        cases=self.payload["cases"],
                    )
                )
            )
            output = root / "run"
            self.assertEqual(
                reproduce_main(
                    [
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                        "--skip-audit",
                        "--cases",
                        "base",
                        "--batches",
                        "1",
                        "2",
                        "--seed",
                        "7",
                        "--warmup-steps",
                        "1",
                        "--measurement-seconds",
                        "0.000000001",
                    ]
                ),
                0,
            )
            payload = json.loads((output / "summary.json").read_text())
            self.assertEqual(payload["workload"]["seed"], 7)
            self.assertEqual(payload["run_settings"]["client_counts"], [1, 2])
            self.assertEqual(payload["validation"]["grid_coverage"], "declared")
            self.assertEqual([row["batch"] for row in payload["results"]], [1, 2])
            with (output / "points.csv").open() as stream:
                self.assertIn("output_tokens_per_s", csv.DictReader(stream).fieldnames)
