"""Native result audits reject unit, capacity, transfer and frontier corruption."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import unittest

from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.native_simulation import point
from scripts.lpx.simulator import pareto
from scripts.lpx.validation import power_snapshot, validate_summary


class LPXResultAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        base = cls.hardware.model("gpt-oss-2t-proxy")
        model = replace(
            base,
            hidden_size=32,
            intermediate_size=64,
            num_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                base.moe, num_experts=8, num_moe_layers=4, moe_intermediate_size=64
            ),
        )
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
            run_settings={"a_replicas": 1},
            normalization_tokens_per_s_per_mw=baseline,
            results=rows,
            frontiers={
                mode: pareto([row for row in rows if row["mode"] == mode])
                for mode in ("rubin", "draft", "af", "pd")
            },
        )

    def test_real_native_points_and_explicit_older_catalog_audit(self):
        report = validate_summary(json.loads(json.dumps(self.payload)))
        self.assertEqual(report["feasible"], 2)
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
