"""The native experiment runs every standard exporter boundary contract."""

import importlib.util
import json
import unittest
from unittest.mock import patch

from scripts import af_probe_rapidjson, deepseek_v3_runtime
from tests.af.trace import test_export


@unittest.skipUnless(
    importlib.util.find_spec("rapidjson"), "optional RapidJSON unavailable"
)
class RapidJSONExportTest(test_export.TraceExportTest):
    def run(self, result=None):
        with (
            patch.object(
                test_export, "write_chrome_trace", af_probe_rapidjson.write_chrome_trace
            ),
            patch.object(
                deepseek_v3_runtime,
                "write_chrome_trace",
                af_probe_rapidjson.write_chrome_trace,
            ),
        ):
            return super().run(result)

    def test_nonfinite_numbers_integer_keys_and_unsupported_values(self):
        encoder = af_probe_rapidjson.CompatibleEncoder()
        values = (
            {"number": float("nan")},
            {"number": float("inf")},
            {"number": float("-inf")},
            {"number": 2**100},
            {"mapping": {True: 1, None: 2, 1.5: 3}},
            {"tuple": (1, 2)},
        )
        for value in values:
            expected = json.dumps(json.loads(json.dumps(value)), sort_keys=True)
            actual = json.dumps(json.loads(encoder.encode(value)), sort_keys=True)
            self.assertEqual(expected, actual)
        for value in ({"mapping": {("x",): 1}}, {"value": {1, 2}}, {"value": b"bytes"}):
            with self.assertRaises(TypeError):
                encoder.encode(value)
