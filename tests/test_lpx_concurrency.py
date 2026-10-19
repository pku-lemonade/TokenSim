"""Concurrency probes respect physical KV limits and measure only decode work."""

from dataclasses import replace
from pathlib import Path
import unittest

from TokenSim.af.runtime.admission import ResidentKV, rank_kv_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.concurrency import cohort_loads, decode_probe, resident_capacity
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration


class LPXConcurrencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        base = cls.hardware.model("gpt-oss-2t-proxy")
        cls.model = replace(
            base,
            num_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                base.moe, num_experts=8, num_moe_layers=2, moe_intermediate_size=64
            ),
        )
        cls.layout = Layout(a_tp=2, a_pp=2, f_ep=2, f_pp=2)

    def test_cohort_assignment_includes_partial_last_cohort(self):
        self.assertEqual(cohort_loads(5, 2, 2), [3, 2])
        self.assertEqual(cohort_loads(9, 4, 3), [4, 4, 1])
        with self.assertRaises(ConfigurationError):
            cohort_loads(5, 0, 2)

    def test_capacity_matches_actual_admission_at_unequal_replica_limits(self):
        runtime = configuration(
            self.hardware, self.model, self.layout, namespace="capacity", a_dp=2
        )
        budgets = {}
        for owner in runtime.plan.request_owners:
            for binding in runtime.plan.bindings_for_instance(
                owner.domain, owner.instance
            ):
                budgets[binding.physical_identity] = rank_kv_bytes(
                    runtime.plan, binding, 128
                ) * (3 if owner.dp_rank == 0 else 5)
        runtime.plan = replace(
            runtime.plan,
            physical_states=tuple(
                replace(
                    state,
                    capacity_bytes=state.model_bytes
                    + state.protocol_buffer_bytes
                    + budgets.get(state.physical_identity, 0),
                )
                for state in runtime.plan.physical_states
            ),
        )
        owners = sorted(
            runtime.plan.request_owners,
            key=lambda o: o.dp_rank,
        )
        for micro, expected in ((1, 6), (2, 5)):
            with self.subTest(micro=micro):
                capacity = resident_capacity(runtime, 128, micro_batch=micro)
                self.assertEqual(capacity["maximum_resident_clients"], expected)
                ledger = ResidentKV(runtime.plan)
                for request_id in range(expected):
                    owner = owners[(request_id // micro) % len(owners)]
                    ledger.admit(
                        request_id,
                        owner.domain,
                        owner.instance,
                        maximum_context=128,
                        current_context=0,
                    )
                owner = owners[(expected // micro) % len(owners)]
                with self.assertRaisesRegex(
                    ConfigurationError, "resident KV does not fit"
                ):
                    ledger.admit(
                        expected,
                        owner.domain,
                        owner.instance,
                        maximum_context=128,
                        current_context=0,
                    )

    def test_decode_counts_exclude_warmup_and_all_cohorts_progress(self):
        runtime = configuration(
            self.hardware, self.model, self.layout, namespace="probe", micro_batch=2
        )
        settings = dict(micro_batch=2, context=16, warmup_steps=1, steps=3)
        row = decode_probe(runtime, 5, **settings)
        self.assertEqual(row, decode_probe(runtime, 5, **settings))
        self.assertEqual(row["cohorts"], 3)
        self.assertGreaterEqual(row["decode_tokens"], 15)
        self.assertGreaterEqual(min(row["completed_steps_per_cohort"]), 3)
        self.assertEqual(row["memory"]["requests"], 5)
        self.assertGreater(row["measurement_s"], 0)
        self.assertLessEqual(row["maximum_resource_utilization"], 1 + 1e-9)
        self.assertLessEqual(
            row["mean_compute_utilization"], row["maximum_compute_utilization"]
        )
        self.assertEqual(row["metric_scope"], "resident_decode_only")
        with self.assertRaises(ConfigurationError):
            decode_probe(runtime, 5, context=16, steps=0)
