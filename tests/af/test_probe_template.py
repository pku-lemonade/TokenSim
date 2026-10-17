"""Template programs retain shared objects and exact dense/MoE event values."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.trace.export import write_chrome_trace
from scripts.af_probe_template import installed
from tests.af.support.profiles import NativeFixture
from tests.support.hardware import make_model


class TemplateProgramTest(NativeFixture, unittest.TestCase):
    def test_dense_and_moe_rows_and_complete_exports(self):
        for model in (make_model(num_layers=2), self.model):
            results = []
            for optimized in (False, True):
                with installed() if optimized else nullcontext():
                    config = build_model_runtime(
                        deepcopy(self.base.doc),
                        hardware=self.base.hardware,
                        model=model,
                    )
                    executor, rows, _, _ = simulate_decode(
                        config, 2, 16, steps=24, fallback="analytical_only"
                    )
                    try:
                        self.assertGreater(executor.plan_builder.template_cache.hits, 0)
                        with TemporaryDirectory() as directory:
                            path = write_chrome_trace(
                                Path(directory) / "trace.json", executor.trace.events
                            )
                            results.append((rows, path.read_bytes()))
                    finally:
                        executor.trace.close()
            self.assertEqual(results[0], results[1])
