"""Online accounting matches full history and releases completed iteration data."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.serving.metrics import ResourceMetrics, steady_metrics
from TokenSim.af.serving.service import AFService
from TokenSim.af.serving.history import IndexedEventHistory
from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.config import Case, Layout, Workload
from scripts.lpx.native_config import configuration
from scripts.lpx.native_simulation import point
from tests.af.trace.test_metrics import event


class ResourceWindowTest(unittest.TestCase):
    def test_online_window_matches_history_with_late_hop_completion_and_drain(self):
        rows = [
            event("warm", 0, 1, resource="worker:0"),
            event("wire2", 1, 2, resource="wire"),
            event("compute", 2, 3, ready=1, resource="worker:0"),
            event("wire1", 0, 4, resource="wire", resource_occupancy_end_at=1),
            event("wire3", 2, 7, resource="wire", resource_occupancy_end_at=3),
            event("drain", 6, 8, resource="worker:0"),
        ]
        window = {}
        resource = ResourceMetrics(window)
        resource.observe(rows[0])
        window["start"] = 1.5
        resource.observe(rows[1])
        resource.observe(rows[2])
        window["end"] = 3
        for row in rows[3:]:
            resource.observe(row)
        self.assertEqual(
            resource.summary([], include_end=True),
            steady_metrics([], rows, start=1.5, end=3, include_end=True),
        )

    def test_cross_iteration_overlap_fails_even_with_low_average_utilization(self):
        rows = [
            event("one", 0, 1, resource="worker:0"),
            replace(event("two", 0.5, 1.5, resource="worker:0"), iteration_id=1),
        ]
        with self.assertRaisesRegex(SimulationStateError, "capacity=1 exceeded"):
            steady_metrics([], rows, start=0, end=10)
        self.assertEqual(
            steady_metrics([], rows, start=0, end=10, capacities={"worker:0": 2})[
                "resource_utilization"
            ]["worker:0"],
            0.1,
        )


class StreamingServingTest(unittest.TestCase):
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
        cls.gpu = Layout(a_tp=2, a_pp=2, f_ep=2, f_pp=2)

    def test_complete_four_mode_results_match_retained_event_history(self):
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
        for case in cases:
            for micro in (1, 2):
                with self.subTest(mode=case.mode, micro=micro):
                    settings = dict(
                        micro_batch=micro, warmup_steps=1, measurement_seconds=0.0002
                    )
                    retained = point(
                        self.hardware, self.model, self.workload, case, 5, **settings
                    )
                    online = point(
                        self.hardware,
                        self.model,
                        self.workload,
                        case,
                        5,
                        stream_metrics=True,
                        **settings,
                    )
                    self.assertEqual(online, retained)
                    completed = [
                        min(
                            turn["end"]
                            for turn in retained["turns"]
                            if not turn["warmup"]
                            and turn["completed"]
                            and turn["cohort"] == cohort
                        )
                        for cohort in range(-(-5 // micro))
                    ]
                    expected_end = max(
                        retained["steady_state"]["start_s"]
                        + settings["measurement_seconds"],
                        *completed,
                    )
                    self.assertEqual(retained["steady_state"]["end_s"], expected_end)

    def test_completed_trace_storage_is_bounded_during_repeated_serving(self):
        runtime = configuration(self.hardware, self.model, self.gpu, namespace="stream")
        env = simpy.Environment()
        resource = ResourceMetrics(dict(start=0))
        service = AFService(
            env, runtime, retain_trace=False, metric_sink=resource.observe
        )
        sessions = [
            service.admit(i, 0, maximum_context=128, current_context=16)
            for i in range(3)
        ]

        def stream(session):
            for _ in range(16):
                yield from service.forward([session])
                session.generated += 1

        for session in sessions:
            env.process(stream(session))
        env.run()
        history = service.trace.events
        self.assertIsInstance(history, IndexedEventHistory)
        self.assertEqual(history.retained_count, 0)
        self.assertFalse(service.trace.iteration_event_indices)
        self.assertLess(history.peak_retained, len(history) / 4)
        self.assertEqual(len(service.completed_passes), 48)
        with self.assertRaises(ConfigurationError):
            AFService(env, runtime, metric_sink=resource.observe)

    def test_short_warmup_turns_restore_the_resident_suffix_snapshot(self):
        contexts = []
        forward = AFService.forward

        def observe(service, sessions, **kwargs):
            if kwargs.get("phase", "decode") == "decode":
                contexts.append(sessions[0].cached_context)
            return (yield from forward(service, sessions, **kwargs))

        with patch.object(AFService, "forward", observe):
            result = point(
                self.hardware,
                self.model,
                replace(self.workload, output_tokens=2),
                Case("short", "rubin", self.gpu),
                1,
                warmup_steps=3,
                measurement_seconds=1e-9,
                stream_metrics=True,
            )
        self.assertEqual(contexts, [18, 18, 18, 18])
        self.assertEqual(result["steady_state"]["output_tokens"], 2)
