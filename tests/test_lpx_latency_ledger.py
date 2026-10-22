"""Accounting checks for the isolated LPX latency audit."""

from collections import Counter
from pathlib import Path
import unittest

from TokenSim.af.providers.traffic import TransferPlanMixin
from TokenSim.af.providers.transfer import TransferEstimateMixin
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.hardware import HardwareContext
from scripts.audit_lpx_latency import bridge, patched, VARIANTS
from scripts.lpx.latency_ledger import (
    WeightedLatency,
    build,
    composed,
    native,
    path_ledger,
    priced_operators,
    identity,
)


class LPXLatencyLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )

    def test_weighted_queries_preserve_grouped_layer_multiplicity(self):
        a = WeightedLatency(2, Counter(a=1))
        b = WeightedLatency(3, Counter(b=1))
        seconds = (3 * a + sum([a, b]) + 7) * 1e-6
        self.assertAlmostEqual(float(seconds), 18e-6)
        self.assertAlmostEqual(seconds.weights["a"], 4e-6)
        self.assertAlmostEqual(seconds.weights["b"], 1e-6)
        with self.assertRaises(TypeError):
            a * b

    def test_unlinked_submit_and_resource_queue_are_accounted_once(self):
        runtime = build(self.hardware)
        key = dict(op_name="rmsnorm", dtype="bf16", num_tokens=1, hidden_size=5376)
        budget, operators = priced_operators(
            runtime, Counter({identity("elementwise", key): 1})
        )
        duration = budget["service_us"] * 1e-6

        def event(node, operation, ready, start, end, parents=(), components=()):
            return AFTraceEvent(
                iteration_id=0,
                block_id=0,
                layer_id=0,
                phase="decode",
                event_type="compute",
                operation=operation,
                resource_id="test",
                ready_at=ready,
                start_at=start,
                end_at=end,
                queue_duration=start - ready,
                service_duration=end - start,
                request_count=1,
                token_count=1,
                source_id="test",
                node_id=node,
                predecessor_ids=parents,
                details={"components": components},
            )

        compute = event(
            "compute",
            "compute",
            4e-6,
            5e-6,
            5e-6 + duration,
            components=[
                dict(
                    name="rmsnorm",
                    table="elementwise",
                    key=key,
                    latency_us=duration * 1e6,
                )
            ],
        )
        transfer = event(
            "transfer",
            "transfer",
            compute.end_at,
            compute.end_at,
            compute.end_at + 2e-6,
            parents=("compute",),
        )
        submit = event("submit", "submit", 0, 0, 4e-6)
        ledger = path_ledger(runtime, [submit, compute, transfer])
        self.assertAlmostEqual(ledger["path_readiness_gap_us"]["compute"], 4)
        self.assertAlmostEqual(ledger["path_queue_us"]["compute"], 1)
        self.assertAlmostEqual(ledger["accounted_us"], 7 + budget["service_us"])
        self.assertEqual(ledger["operator_budget"]["operators"], 1)

    def test_native_composition_bridge_and_patches_restore_defaults(self):
        original = TransferPlanMixin._collective
        transfer = TransferEstimateMixin.estimate
        isolated = composed(build(self.hardware), 10750)
        points = {}
        for variant in VARIANTS:
            runtime = build(self.hardware)
            with patched(runtime, variant):
                points[variant] = native(runtime, 10750)
            self.assertIs(TransferPlanMixin._collective, original)
            self.assertIs(TransferEstimateMixin.estimate, transfer)
        result = bridge(isolated, points)
        self.assertEqual(isolated["operator_budget"]["operators"], 16 * 60 + 2)
        self.assertEqual(points["default"]["operator_budget"]["operators"], 16 * 60 + 4)
        for point in points.values():
            self.assertEqual(point["event_counts"], points["default"]["event_counts"])
        handoff = result["differences_us"]["internal_af_copy_and_queue_us"]
        self.assertGreater(handoff, 0)
        self.assertGreater(result["differences_us"]["residual_gathers_us"], 0)
        repeated = native(build(self.hardware), 10750)
        self.assertEqual(repeated["latency_us"], points["default"]["latency_us"])


if __name__ == "__main__":
    unittest.main()
