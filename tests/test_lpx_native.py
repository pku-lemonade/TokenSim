"""Native serving metrics, rank inventories and speculative rollback."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import simpy

from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.native_simulation import point
from scripts.lpx.native_config import configuration
from TokenSim.af.serving.service import AFService
from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.factory import build_model_runtime


class NativeLPXTest(unittest.TestCase):
    def test_external_draft_replicas_distribute_clients_and_charge_inventory(self):
        draft_layout = replace(self.base, a_device="groq_lp30", f_device="groq_lp30")
        case = Case(
            "draft-dp2", "draft", self.base, draft=draft_layout, draft_a_replicas=2
        )
        result = point(
            self.hardware,
            self.model,
            self.workload,
            case,
            4,
            draft_spec=self.model,
            warmup_steps=1,
            measurement_seconds=1e-9,
        )
        self.assertEqual(result["inventory"], {"rubin": 2, "groq_lp30": 4})
        self.assertEqual(result["inventory"], case.inventory())
        self.assertEqual(result["services"]["draft"]["memory"]["requests"], 4)
        self.assertEqual(
            {
                event["source"]
                for event in result["control_events"]
                if event["source"].startswith("draft-a:")
            },
            {"draft-a:0", "draft-a:2"},
        )

    def test_host_collection_policy_preserves_native_events(self):
        case = Case("af", "af", self.lpu, self.base)
        settings = dict(warmup_steps=1, measurement_seconds=1e-9, trace=True)
        original = point.__wrapped__(
            self.hardware, self.model, self.workload, case, 2, **settings
        )
        amortized = point(self.hardware, self.model, self.workload, case, 2, **settings)
        self.assertEqual(original, amortized)

    def test_return_routes_exclude_stages_without_a_shared_layer(self):
        from TokenSim.config.cache_config import stage_layer_range
        from TokenSim.errors import ConfigurationError

        model = replace(
            self.model, num_layers=4, moe=replace(self.model.moe, num_moe_layers=4)
        )
        layout = Layout(
            a_tp=2, a_pp=4, f_device="groq_lp30", f_ep=4, f_pp=2, colocated=False
        )
        runtime = configuration(self.hardware, model, layout, namespace="sparse")
        plan = runtime.plan
        self.assertEqual(len(plan.partial_destinations), 32)
        destination = plan.partial_destinations[0]
        for route in plan.partial_destinations:
            source = plan.binding(route.source_domain, route.source_rank)
            target = plan.binding(route.target_domain, route.target_rank)
            self.assertTrue(
                set(stage_layer_range(4, 2, source.pp_rank)).intersection(
                    stage_layer_range(4, 4, target.pp_rank)
                )
            )
        invalid = replace(destination, target_rank=4)
        broken = replace(
            plan, partial_destinations=(invalid, *plan.partial_destinations[1:])
        )
        with self.assertRaisesRegex(ConfigurationError, "share a model layer"):
            broken.validate()

    def test_reference_planning_preserves_complete_native_point_results(self):
        case = Case("base", "rubin", self.base)
        settings = dict(
            warmup_steps=1,
            measurement_seconds=1e-9,
            trace=True,
            dag_template_cache=False,
        )
        reference = point(self.hardware, self.model, self.workload, case, 1, **settings)

        def cached(doc, **kwargs):
            doc["af"]["dag_template_cache"] = True
            return build_model_runtime(doc, **kwargs)

        with patch("scripts.lpx.native_config.build_model_runtime", cached):
            compiled = point(
                self.hardware, self.model, self.workload, case, 1, **settings
            )
        self.assertEqual(reference, compiled)

    def test_intermediate_prefill_extends_kv_without_unneeded_logits(self):
        runtime = configuration(self.hardware, self.model, self.base, namespace="chunk")
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        env.process(
            service.forward(
                [session], phase="prefill", query_tokens=2, emit_logits=False
            )
        )
        env.run()
        self.assertEqual(session.cached_context, 18)
        self.assertFalse(
            any(e.operation in {"final_norm", "lm_head"} for e in service.trace.events)
        )
        env.process(service.forward([session], phase="prefill", query_tokens=2))
        env.run()
        self.assertEqual(session.cached_context, 20)
        forwards = [
            e for e in service.trace.events if e.operation == "complete_model_prefill"
        ]
        self.assertEqual([e.details["logits_rows"] for e in forwards], [0, 1])
        self.assertEqual(service.executor._next_iteration_id, 0)
        self.assertFalse(
            any(e.operation in {"af_a_to_f", "af_f_to_a"} for e in service.trace.events)
        )

    def test_prefill_bypasses_af_submission_and_rejects_split_pools(self):
        from TokenSim.errors import ConfigurationError

        runtime = configuration(self.hardware, self.model, self.base, namespace="full")
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        with patch.object(
            service.executor,
            "submit",
            side_effect=AssertionError("AF prefill submitted"),
        ):
            env.run(
                until=env.process(
                    service.forward([session], phase="prefill", query_tokens=2)
                )
            )
        self.assertEqual(session.cached_context, 18)
        split = configuration(self.hardware, self.model, self.lpu, namespace="split")
        other = AFService(env, split)
        session = other.admit(1, 0, maximum_context=24, current_context=16)
        with self.assertRaisesRegex(ConfigurationError, "complete-model pool"):
            env.run(
                until=env.process(
                    other.forward([session], phase="prefill", query_tokens=2)
                )
            )
        self.assertEqual(session.cached_context, 16)
        self.assertFalse(other.trace.events)
        self.assertEqual(other.executor._next_iteration_id, 0)

    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        base = cls.hardware.model("gpt-oss-2t-expert-proxy")
        cls.model = replace(
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
            draft_spec=self.model,
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
            Case(
                "draft",
                "draft",
                self.base,
                draft=replace(self.lpu, a_pp=4, f_pp=4, f_ep=2, colocated=True),
            ),
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
        case = Case(
            "draft",
            "draft",
            self.base,
            draft=replace(self.lpu, a_pp=4, f_pp=4, f_ep=2, colocated=True),
        )
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
            transfer_occupancy="wire",
        )
        services = [AFService(env, r, communication=endpoints) for r in runtimes]
        sessions = [
            s.admit(0, 0, maximum_context=24, current_context=16) for s in services
        ]
        for service, session in zip(services, sessions):
            env.process(service.forward([session]))
        env.run()
        self.assertEqual([s.cached_context for s in sessions], [17, 17])

    def test_complete_turn_counts_exclude_warm_tokens_and_include_final_verification(
        self,
    ):
        for acceptance in (0, 1):
            case = Case(
                "draft",
                "draft",
                self.base,
                draft=replace(self.lpu, a_pp=4, f_pp=4, f_ep=2, colocated=True),
            )
            result = point(
                self.hardware,
                self.model,
                replace(self.workload, acceptance_probability=acceptance),
                case,
                1,
                warmup_steps=1,
                measurement_seconds=1e-9,
                draft_spec=self.model,
            )
            self.assertEqual(result["steady_state"]["output_tokens"], 8)
            self.assertEqual(result["steady_state"]["decode_tokens"], 7)

    def test_pipeline_concurrency_increases_factory_output_on_the_same_inventory(self):
        case = Case("pipeline", "rubin", replace(self.base, a_pp=2, f_pp=2))
        settings = dict(warmup_steps=1, measurement_seconds=1e-9)
        one = point(self.hardware, self.model, self.workload, case, 1, **settings)
        two = point(self.hardware, self.model, self.workload, case, 2, **settings)
        self.assertEqual(one["inventory"], two["inventory"])
        self.assertEqual(one["power_watts"], two["power_watts"])
        self.assertGreater(two["output_tokens_per_s"], one["output_tokens_per_s"])
        for rank, size in one["services"]["target"]["memory"][
            "reserved_peak_kv_bytes"
        ].items():
            self.assertEqual(
                two["services"]["target"]["memory"]["reserved_peak_kv_bytes"][rank],
                2 * size,
            )

    def test_warm_snapshot_preserves_measured_work_and_draft_prefills_overlap(self):
        case = Case("base", "rubin", self.base)
        settings = dict(warmup_steps=1, measurement_seconds=1e-9)
        resident = point(self.hardware, self.model, self.workload, case, 1, **settings)
        full = point(
            self.hardware,
            self.model,
            self.workload,
            case,
            1,
            warmup_mode="prefill_and_decode",
            **settings,
        )
        for key in ("tokens_per_s_per_user", "output_tokens_per_s"):
            self.assertAlmostEqual(resident[key], full[key], places=7)
        self.assertEqual(resident["steady_state"]["output_tokens"], 8)
        self.assertEqual(resident["timings"]["drain_s"], 0)
        draft = point(
            self.hardware,
            self.model,
            self.workload,
            Case(
                "draft",
                "draft",
                self.base,
                draft=replace(self.lpu, a_pp=4, f_pp=4, f_ep=2, colocated=True),
            ),
            1,
            trace=True,
            draft_spec=self.model,
            **settings,
        )
        starts = {
            name: min(
                e["start_at"]
                for e in draft["trace"]
                if e["phase"] == "prefill"
                and (e["source_domain"] or "").startswith(name)
            )
            for name in ("target", "draft")
        }
        self.assertEqual(starts["target"], starts["draft"])
