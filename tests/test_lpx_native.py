"""Native serving metrics, rank inventories and speculative rollback."""

from dataclasses import replace
from pathlib import Path
import unittest
import simpy

from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.native_simulation import point
from scripts.lpx.native_config import configuration
from TokenSim.af.serving.service import AFService
from TokenSim.af.communication.endpoints import EndpointResources


class NativeLPXTest(unittest.TestCase):
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
                base.moe, num_experts=8, num_moe_layers=4, moe_intermediate_size=64
            ),
        )
        cls.workload = Workload(
            cached_tokens=16, new_tokens=2, output_tokens=8, prefill_chunk=2
        )
        cls.base = Layout(a_tp=2, f_ep=2)
        cls.lpu = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=2,
            a_pp=4,
            f_ep=4,
            f_pp=4,
            colocated=False,
        )

    def run_point(self, case, **kwargs):
        workload = kwargs.pop("workload", self.workload)
        return point(
            self.hardware,
            self.model,
            workload,
            case,
            2,
            warmup_steps=1,
            measurement_seconds=0.0002,
            **kwargs,
        )

    def test_four_modes_have_native_steady_output_and_bounded_utilization(self):
        cases = [
            Case("rubin", "rubin", self.base),
            Case(
                "af",
                "af",
                replace(self.base, f_device="groq_lp30", f_ep=4, colocated=False),
                self.base,
            ),
            Case("pd", "pd", self.lpu, self.base),
            Case("draft", "draft", self.base, draft=self.lpu),
        ]
        for case in cases:
            with self.subTest(mode=case.mode):
                result = self.run_point(case)
                self.assertEqual(result["native_executor"], "ModelAFExecutor")
                self.assertGreater(result["steady_state"]["decode_tokens"], 0)
                self.assertGreater(result["timings"]["warmup_s"], 0)
                for value in result["steady_state"]["resource_utilization"].values():
                    self.assertLessEqual(value, 1 + 1e-9)
                repeat = self.run_point(case)
                self.assertEqual(result, repeat)

    def test_replicas_charge_physical_inventory_and_power_only_changes_efficiency(self):
        case = Case("rubin", "rubin", self.base)
        low = self.run_point(case, a_dp=2, power_overrides={"rubin": 1000})
        high = self.run_point(case, a_dp=2, power_overrides={"rubin": 2000})
        self.assertEqual(low["inventory"], {"rubin": 4})
        self.assertEqual(low["tokens_per_s_per_user"], high["tokens_per_s_per_user"])
        self.assertEqual(low["tokens_per_s_per_mw"], 2 * high["tokens_per_s_per_mw"])

    def test_graph_submission_and_full_verification_are_charged(self):
        case = Case("draft", "draft", self.base, draft=self.lpu)
        for acceptance in (0, 1):
            result = self.run_point(
                case,
                trace=True,
                workload=replace(self.workload, acceptance_probability=acceptance),
            )
            self.assertTrue(
                any(e["operation"] == "cuda_graph_submit" for e in result["trace"])
            )
            # Every target decode pass executes all gamma+1 positions,
            # regardless of the number subsequently committed.
            verification = [
                e
                for e in result["trace"]
                if e["phase"] == "decode"
                and e["operation"] == "lm_head"
                and e["source_domain"].startswith("target")
            ]
            self.assertTrue(verification)
            self.assertEqual({e["token_count"] for e in verification}, {5})
            cycles = result["verification_events"]
            if acceptance == 0:
                self.assertEqual({row["draft_catchup_tokens"] for row in cycles}, {0})
            else:
                self.assertTrue(any(row["draft_catchup_tokens"] == 1 for row in cycles))
            for service in result["services"].values():
                memory = service["memory"]
                for rank, current in memory["resident_current_kv_bytes"].items():
                    self.assertLessEqual(
                        current, memory["reserved_peak_kv_bytes"][rank]
                    )

    def test_compact_history_matches_full_trace_metrics_and_namespaces_pools(self):
        case = Case(
            "af",
            "af",
            replace(self.base, f_device="groq_lp30", f_ep=4, colocated=False),
            self.base,
        )
        compact = self.run_point(case)
        full = self.run_point(case, trace=True)
        del compact["trace"], full["trace"]
        self.assertEqual(compact, full)
        keys = compact["steady_state"]["resource_utilization"]
        self.assertTrue(any(key.startswith("target/worker:") for key in keys))
        self.assertTrue(any(key.startswith("prefill/worker:") for key in keys))

    def test_simultaneous_services_with_equal_local_node_ids_share_endpoint_scheduler(
        self,
    ):
        runtimes = [
            configuration(self.hardware, self.model, self.base, namespace=name)
            for name in ("one", "two")
        ]
        env = simpy.Environment()
        endpoints = EndpointResources(
            env,
            {key: value for r in runtimes for key, value in r.cluster.networks.items()},
            [w.physical_worker_ref.identity for r in runtimes for w in r.workers],
        )
        services = [AFService(env, r, communication=endpoints) for r in runtimes]
        sessions = [
            s.admit(0, 0, maximum_context=24, current_context=16) for s in services
        ]
        for service, session in zip(services, sessions):
            env.process(service.forward([session]))
        env.run()
        self.assertEqual([s.cached_context for s in sessions], [17, 17])
