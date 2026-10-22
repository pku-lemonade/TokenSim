"""Native rank ownership, pipeline returns and complete-model prefill."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx_runtime import Layout, configuration
from tests.support.lpx import tiny_moe


class NativeLPXTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = tiny_moe(cls.hardware)
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

    def test_unsupported_colocated_hybrids_are_rejected_before_remapping(self):
        for layout, replicas in (
            (Layout(a_tp=4, f_tp=2, f_ep=2), 1),
            (Layout(a_tp=4, f_tp=2, f_ep=2), 2),
            (Layout(a_tp=2, f_tp=2, f_ep=1), 2),
        ):
            with (
                self.subTest(replicas=replicas),
                patch("scripts.lpx_runtime.build_model_runtime") as build,
                self.assertRaisesRegex(ConfigurationError, "hybrid expert TP/EP"),
            ):
                configuration(
                    self.hardware,
                    self.model,
                    layout,
                    namespace="mixed",
                    a_dp=replicas,
                )
            build.assert_not_called()

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

    def prefill(self, device, mode, overhead):
        f = self
        layout = Layout(a_device=device, f_device=device, a_tp=2, f_ep=2)
        runtime = configuration(
            f.hardware,
            f.model,
            layout,
            namespace="settings",
            execution_mode=mode,
            graph_kernel_us=overhead,
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        env.run(
            until=env.process(
                service.forward([session], phase="prefill", query_tokens=2)
            )
        )
        return env.now, service

    def test_gpu_prefill_honors_graph_kernel_assumption_without_mutating_catalog(self):
        original = self.hardware.device("rubin").analytical["kernel_launch_us"]
        eager, eager_service = self.prefill("rubin", "eager", 0.1)
        graph, graph_service = self.prefill("rubin", "cuda_graph", 0.1)
        self.assertLess(graph, eager)
        for service, expected in ((eager_service, original), (graph_service, 0.1)):
            self.assertTrue(service.prefill.backends)
            for backend in service.prefill.backends.values():
                self.assertEqual(
                    backend.device.analytical["kernel_launch_us"], expected
                )
            self.assertEqual(service.sessions[0].cached_context, 18)
        self.assertEqual(
            self.hardware.device("rubin").analytical["kernel_launch_us"],
            original,
        )

    def test_gpu_graph_setting_does_not_override_lpu_execution_costs(self):
        eager, _ = self.prefill("groq_lp30", "eager", 0.1)
        graph, service = self.prefill("groq_lp30", "cuda_graph", 0.1)
        self.assertEqual(eager, graph)
        expected = self.hardware.device("groq_lp30").analytical["kernel_launch_us"]
        self.assertTrue(
            all(
                b.device.analytical["kernel_launch_us"] == expected
                for b in service.prefill.backends.values()
            )
        )

    def test_colocated_expert_tp_survives_prefill_and_decode(self):
        runtime = configuration(
            self.hardware, self.model, Layout(a_tp=2, f_tp=2, f_ep=1), namespace="pure"
        )
        domain = runtime.cluster.af_domains[1].name
        self.assertEqual(
            runtime.plan.domain_runtime(domain).expert_tensor_parallel_size, 2
        )
        self.assertEqual(len(runtime.plan.expert_execution(domain, 0, 0)), 2)
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        for phase, queries in (("prefill", 2), ("decode", 1)):
            env.run(
                until=env.process(
                    service.forward([session], phase=phase, query_tokens=queries)
                )
            )
        self.assertEqual(session.cached_context, 19)
        self.assertTrue(any(e.layer_type == "moe" for e in service.trace.events))
        service.release(session)
        self.assertFalse(service.memory.leases)
