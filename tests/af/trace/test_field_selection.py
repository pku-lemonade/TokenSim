"""Trace configuration files and explicit event fields share export semantics."""

import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.errors import ConfigurationError
from scripts import run_deepseek_mla as cli
from scripts.af_trace_options import trace_configuration
from tests.af.trace.test_metrics import event


class TraceFieldSelectionTest(unittest.TestCase):
    def test_explicit_fields_preserve_chrome_envelope_and_exact_values(self):
        row = event("node", 1, 2, note="selected detail")
        for names in ([], ["node_id", "details", "predecessor_ids"]):
            with self.subTest(names=names), TemporaryDirectory() as directory:
                path = Path(directory) / "trace.json"
                config = AFTraceConfig(fields=names)
                write_chrome_trace(path, [row], trace_config=config)
                exported = json.loads(path.read_text())
                item = exported["traceEvents"][0]
                self.assertEqual(set(item["args"]), set(names))
                self.assertEqual(item["ts"], 1e6)
                self.assertEqual(item["dur"], 1e6)
                self.assertEqual(
                    item["name"], row.stage or row.operation or row.event_type
                )
                if names:
                    self.assertEqual(item["args"]["details"], row.details)
                else:
                    self.assertEqual(
                        config.describe()["dependency_scope"], "predecessor_ids_omitted"
                    )

    def test_invalid_fields_and_config_objects_fail_early(self):
        for values in (
            {"fields": "node_id"},
            {"fields": ["unknown"]},
            {"fields": ["node_id"], "detail": "timing"},
            {"fields": [], "mode": "off"},
            {"unknown": True},
            [],
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                AFTraceConfig.from_mapping(values)

    def test_json_file_and_cli_override_merge_without_resetting_other_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace-config.json"
            path.write_text(
                json.dumps(
                    dict(event_types=["transfer"], iterations=[0], fields=["node_id"])
                )
            )
            args = cli.build_parser().parse_args(
                [
                    "--trace-config",
                    str(path),
                    "--trace-iterations",
                    "2",
                    "--no-trace-validation",
                ]
            )
            options = trace_configuration(args)
            self.assertEqual(options.event_types, ("transfer",))
            self.assertEqual(options.iterations, (2,))
            self.assertEqual(options.fields, ("node_id",))
            self.assertFalse(options.validate)

    def test_invalid_files_do_not_create_outputs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace-config.json"
            output = Path(directory) / "output"
            for text in ("[]", "broken JSON", '{"unknown":true}'):
                path.write_text(text)
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    cli.main(["--trace-config", str(path), "--output", str(output)])
                self.assertFalse(output.exists())
