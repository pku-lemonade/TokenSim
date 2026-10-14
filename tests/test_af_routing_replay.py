import copy
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from tests import test_af_host_profile as host
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.errors import ConfigurationError, WorkloadValidationError


class RoutingReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host.HostProfileTest.setUpClass()
        cls.fixture = host.HostProfileTest()

    def configured(self):
        runtime = self.fixture.runtime()
        profile = self.fixture.document(runtime)
        replay = dict(
            schema_version=1,
            runtime_fingerprint=profile["runtime_fingerprint"],
            environment=profile["environment"],
            source="unit-test://routing",
            kind="synthetic_test",
            assignments=[
                dict(
                    phase="decode",
                    decode_step=step,
                    layer_id=layer,
                    request_id=0,
                    token_id=128 + step,
                    top_k_index=k,
                    expert_id=(layer + k) % 4,
                    routing_weight=0.25 if k == 0 else 0.75,
                    a_dp_rank=0,
                    a_tp_rank=0,
                )
                for step in range(2)
                for layer in range(2)
                for k in range(2)
            ],
        )
        doc = copy.deepcopy(runtime.doc)
        doc["af"].update(
            routing_replay=replay, calibration_environment=profile["environment"]
        )
        return doc, runtime

    def run_doc(self, doc, runtime, **options):
        configured = build_model_runtime(
            doc, hardware=runtime.hardware, model=runtime.model
        )
        return simulate_decode(
            configured, 1, 129, steps=2, fallback="analytical_only", **options
        )

    def test_exact_assignments_and_cache_trace_equivalence(self):
        doc, runtime = self.configured()
        executor, rows, _, _ = self.run_doc(doc, runtime)
        routers = [event for event in executor.trace.events if event.stage == "router"]
        self.assertTrue(routers)
        self.assertTrue(
            all(
                event.details["routing_replay"]["kind"] == "synthetic_test"
                for event in routers
            )
        )
        request = SimpleNamespace(
            id=0, is_prefill=False, context_len=129, generation_idx=0
        )
        iteration = executor.submit(0, [request], "decode")
        records = []
        for snapshot in iteration.blocks[0].routing_snapshots:
            self.assertEqual(sum(snapshot.histogram.values()), 2)
            for assignment in snapshot.assignments:
                record = asdict(assignment)
                record.pop("iteration_id")
                records.append(record)
        self.assertEqual(records, doc["af"]["routing_replay"]["assignments"][:4])
        self.assertFalse(hasattr(request, "decode_routing_assignments"))
        doc["af"].update(service_estimate_cache=False, dag_template_cache=False)
        _, without_cache, _, _ = self.run_doc(doc, runtime, trace=False)
        self.assertEqual(rows, without_cache)
        self.assertEqual(executor.routing_replay.describe()["kind"], "synthetic_test")

    def test_missing_duplicate_wrong_token_expert_and_environment_fail(self):
        original, runtime = self.configured()
        edits = [
            lambda r: r["assignments"].pop(),
            lambda r: r["assignments"].append(r["assignments"][0]),
            lambda r: r["assignments"][0].update(token_id=9),
            lambda r: r["assignments"][0].update(expert_id=99),
            lambda r: r.update(environment={}),
            lambda r: r["assignments"].clear(),
            lambda r: r.update(runtime_fingerprint="wrong"),
        ]
        for edit in edits:
            doc = copy.deepcopy(original)
            edit(doc["af"]["routing_replay"])
            with (
                self.subTest(edit=edit),
                self.assertRaises((ConfigurationError, WorkloadValidationError)),
            ):
                self.run_doc(doc, runtime)


if __name__ == "__main__":
    unittest.main()
