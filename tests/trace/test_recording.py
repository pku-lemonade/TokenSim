"""Common storage contracts and invalid timing regression coverage."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.trace.config import TraceConfig
from TokenSim.trace.events import TraceEvent
from TokenSim.trace.formats import (
    make_trace_recorder,
    write_trace,
    convert_trace_to_json,
)
from TokenSim.trace.recorder import TraceRecorder


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

    def test_af_adapter_reuses_finite_timestamp_validation(self):
        from TokenSim.af.trace.metrics import AFTraceEvent
        from TokenSim.af.trace.recorder import AFTraceRecorder

        recorder = AFTraceRecorder()
        recorder.record(AFTraceEvent(**event(ready_at=float("nan")).as_dict()))
        with self.assertRaisesRegex(SimulationStateError, "finite"):
            recorder.validate_invariants(0)
