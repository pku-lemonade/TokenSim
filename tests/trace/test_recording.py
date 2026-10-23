"""Common storage contracts and invalid timing regression coverage."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import simpy

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.trace.config import TraceConfig
from TokenSim.trace.events import TraceEvent
from TokenSim.trace.formats import (
    make_trace_recorder,
    write_trace,
    convert_trace_to_json,
)
from TokenSim.trace.recorder import TraceRecorder
from TokenSim.trace.execution import observe_service
from TokenSim.trace.session import TraceSession


def event(**changes):
    return replace(
        TraceEvent(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            event_type="compute",
            resource_id="gpu0",
            ready_at=0,
            start_at=0,
            end_at=1e-6,
            queue_duration=0,
            service_duration=1e-6,
            request_count=1,
            token_count=1,
            source_id="fixture",
            node_id="node0",
            operation="rmsnorm",
        ),
        **changes,
    )


class SharedRecordingTest(unittest.TestCase):
    def test_eof_roundoff_does_not_mark_a_completed_operator_as_partial(self):
        for excess in (math.ulp(1.0), 0.1):
            with self.subTest(excess=excess):
                session = object.__new__(TraceSession)
                session.engine = SimpleNamespace(
                    af_executor=None, env=SimpleNamespace(now=1.0)
                )
                session.recorder = SimpleNamespace(
                    events=[
                        event(
                            start_at=0.5,
                            end_at=1.0 + excess,
                            service_duration=0.5 + excess,
                        )
                    ]
                )
                session.transfers = SimpleNamespace(partial_events=lambda: ())
                session.pending = {}
                observed = list(session.events())[0]
                self.assertEqual(observed.end_at, 1.0)
                self.assertEqual(observed.incomplete, excess == 0.1)

    def test_service_observation_starts_after_admission_and_releases_on_cancel(self):
        env = simpy.Environment()
        resource = simpy.Resource(env, capacity=1)
        starts = []

        def holder():
            with resource.request() as request:
                yield request
                yield env.timeout(2)

        def service():
            with resource.request() as request:
                yield request
                yield env.timeout(10)

        def observed():
            try:
                yield from observe_service(service(), lambda: starts.append(env.now))
            except simpy.Interrupt:
                pass

        env.process(holder())
        process = env.process(observed())
        env.run(until=3)
        self.assertEqual(starts, [2])
        process.interrupt("cancel")
        env.run()
        self.assertEqual(resource.count, 0)

    def test_shared_export_imports_no_af_modules(self):
        script = "import sys; import TokenSim.trace.formats; assert not any(n.startswith('TokenSim.af') for n in sys.modules)"
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_nonfinite_readiness_is_rejected(self):
        for field in (
            "ready_at",
            "start_at",
            "end_at",
            "service_duration",
            "queue_duration",
        ):
            with self.subTest(field=field):
                recorder = TraceRecorder()
                recorder.record(event(**{field: float("nan")}))
                with self.assertRaisesRegex(SimulationStateError, "finite"):
                    recorder.validate_invariants(0)

    def test_dependencies_and_resource_conflicts_are_checked(self):
        recorder = TraceRecorder()
        recorder.record(event())
        recorder.record(event(node_id="node1", predecessor_ids=("node0",)))
        with self.assertRaisesRegex(SimulationStateError, "not ready"):
            recorder.validate_invariants(0)
        recorder.events[1] = event(node_id="node1")
        with self.assertRaisesRegex(SimulationStateError, "capacity"):
            recorder.validate_invariants(0)

    def test_archive_formats_preserve_common_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            traces = []
            for format in ("json", "archive") + (
                ("msgpack",) if importlib.util.find_spec("msgpack") else ()
            ):
                options = TraceConfig(mode="full")
                recorder = make_trace_recorder(path, options, format)
                recorder.record(event())
                recorder.validate_invariants(0)
                recorder.complete_iteration(0)
                source = write_trace(
                    path / f"{format}.json",
                    recorder.events,
                    trace_config=options,
                    trace_format=format,
                )
                output = path / f"converted-{format}.json"
                convert_trace_to_json(source, output)
                traces.append(json.loads(output.read_text()))
                recorder.close()
            self.assertEqual(traces[0], traces[1])
            for trace in traces[1:]:
                self.assertEqual(traces[0], trace)
            self.assertEqual(traces[0]["traceEvents"][0]["dur"], 1)

    def test_global_defaults_and_boundary_validation(self):
        from TokenSim.af.trace.config import AFTraceConfig
        from TokenSim.config.simulation import SimulationConfig

        self.assertFalse(AFTraceConfig("full", False).validate)
        self.assertEqual(SimulationConfig(4).block_size, 4)
        self.assertFalse(TraceConfig().enabled)
        for settings in (
            {"mode": "bad"},
            {"mode": "off", "output": "x"},
            {"mode": "full", "workers": [True]},
            {"mode": "full", "phases": ["other"]},
            {"unknown": 1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ConfigurationError):
                TraceConfig.from_mapping(settings)

    def test_zero_duration_dependency_cycle_is_rejected(self):
        recorder = TraceRecorder()
        recorder.record(event(end_at=0, service_duration=0, predecessor_ids=("node1",)))
        recorder.record(
            event(
                end_at=0,
                service_duration=0,
                node_id="node1",
                predecessor_ids=("node0",),
            )
        )
        with self.assertRaisesRegex(SimulationStateError, "cycle"):
            recorder.validate_invariants(0)

    def test_late_clock_preserves_small_service_without_false_failure(self):
        recorder = TraceRecorder()
        recorder.record(
            event(ready_at=1e8, start_at=1e8, end_at=1e8 + 1e-6, service_duration=1e-6)
        )
        recorder.validate_invariants(0)

    def test_flow_export_connects_only_visible_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                event(),
                event(
                    iteration_id=1,
                    node_id="node1",
                    ready_at=1e-6,
                    start_at=1e-6,
                    end_at=2e-6,
                    predecessor_ids=("node0",),
                ),
            ]
            path = Path(directory) / "trace.json"
            write_trace(path, rows, {"schema_version": 2})
            data = json.loads(path.read_text())["traceEvents"]
            self.assertEqual(
                [row["ph"] for row in data if row["ph"] in ("s", "f")], ["s", "f"]
            )
            self.assertTrue(all(isinstance(row["tid"], int) for row in data))
            write_trace(
                path,
                rows,
                {"schema_version": 2},
                trace_config=TraceConfig(mode="full", iterations=(1,)),
            )
            data = json.loads(path.read_text())["traceEvents"]
            self.assertEqual([row["ph"] for row in data], ["M", "X"])

    def test_af_adapter_reuses_finite_timestamp_validation(self):
        from TokenSim.af.trace.metrics import AFTraceEvent
        from TokenSim.af.trace.recorder import AFTraceRecorder

        recorder = AFTraceRecorder()
        recorder.record(AFTraceEvent(**event(ready_at=float("nan")).as_dict()))
        with self.assertRaisesRegex(SimulationStateError, "finite"):
            recorder.validate_invariants(0)
