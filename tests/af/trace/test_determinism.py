"""Seeded trace contents and resource metadata survive process hash changes."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.serving.service import AFService
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.moe import RoutingConfig
from tests.af.support.profiles import NativeFixture

ROOT = Path(__file__).resolve().parents[3]


def capture_seeded_traces(seed=42):
    NativeFixture.setUpClass()
    fixture = NativeFixture()
    model = replace(
        fixture.model,
        moe=replace(
            fixture.model.moe,
            routing=RoutingConfig(uniform_selection="random_without_replacement"),
        ),
    )
    observations = {}
    for mode, control in (("eager", False), ("cuda_graph", False), ("eager", True)):
        runtime = build_model_runtime(
            fixture.document(
                compute_execution_mode=mode,
                communication_timing=dict(
                    enabled=True,
                    host_submit_us=2,
                    host_completion_us=1,
                    native_control_enabled=control,
                ),
            ),
            hardware=fixture.base.hardware,
            model=model,
        )
        env = simpy.Environment()
        service = AFService(env, runtime, seed=seed)
        sessions = [
            service.admit(i, 0, maximum_context=160, current_context=128)
            for i in range(8)
        ]

        def stream(group):
            for _ in range(3):
                yield from service.forward(group)
                for session in group:
                    session.generated += 1

        for begin in (0,) if control else (0, 4):
            env.process(stream(sessions[begin : begin + 4]))
        try:
            env.run()
            # Include the real resource mapping in exported metadata so key
            # order is checked as well as event order, dependencies and timing.
            with TemporaryDirectory() as directory:
                output = Path(directory) / "trace.json"
                write_chrome_trace(
                    output, service.trace.events, metadata=service.summary()
                )
                observations[f"{mode}-{control}"] = dict(
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    events=len(service.trace.events),
                    passes=service.completed_passes,
                )
        finally:
            service.trace.close()
    return observations


class TraceDeterminismTest(unittest.TestCase):
    def test_seeded_events_and_metadata_match_across_fresh_processes(self):
        command = (
            "import json; "
            "from tests.af.trace.test_determinism import capture_seeded_traces; "
            "print(json.dumps(capture_seeded_traces()))"
        )
        observations = []
        for hash_seed in ("1", "2", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", command],
                cwd=ROOT,
                env=dict(os.environ, PYTHONHASHSEED=hash_seed),
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            observations.append(json.loads(result.stdout))
        for observation in observations[1:]:
            self.assertEqual(observation, observations[0])

    def test_repeat_ignores_global_random_state_but_uses_explicit_seed(self):
        previous = random.getstate()
        self.addCleanup(random.setstate, previous)
        random.seed(123)
        reference = capture_seeded_traces()
        random.seed(987)
        self.assertEqual(capture_seeded_traces(), reference)
        changed = capture_seeded_traces(seed=43)
        for name, observation in reference.items():
            self.assertGreater(observation["events"], 0)
            self.assertNotEqual(observation["sha256"], changed[name]["sha256"])
