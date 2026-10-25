"""Timeline exports retain gaps, raw activities, clocks and concurrent streams."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from TokenSim.profiler.compare import compare_traces
from TokenSim.profiler.plot import export_timeline
from TokenSim.profiler.timeline import prepare_timeline, pack_tracks
from tests.trace.test_profiler import fixtures


class TimelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        simulated, profile = fixtures(self.root)
        self.report_path = self.root / "comparison/comparison.json"
        self.report = compare_traces(simulated, [profile], self.report_path.parent)

    def test_kernel_names_streams_and_original_gaps_are_retained(self):
        result = prepare_timeline(self.report)
        actual = [bar for bar in result["bars"] if bar.side == "profiled"]
        self.assertEqual([bar.name for bar in actual], ["gemm_kernel", "gemm_kernel"])
        self.assertEqual([bar.track for bar in actual], ["stream 7", "stream 7"])
        self.assertEqual([bar.end_us - bar.start_us for bar in actual], [5, 4])
        self.assertEqual(actual[1].start_us - actual[0].end_us, 2)
        self.assertEqual(actual[0].start_us + actual[0].translation_us, 0)

    def test_unmatched_rank_retains_a_proven_shared_clock(self):
        extra = copy.deepcopy(self.report["matches"][0]["profiled"])
        extra.update(
            id="unmatched-rank",
            rank="1",
            clock="profile:1",
            side="profiled",
            status="unmatched",
            start_us=2008,
            end_us=2013,
        )
        extra["details"]["service_intervals"] = [[2008, 2013]]
        extra["details"]["device_activities"][0].update(start_us=2008, end_us=2013)
        self.report["unmatched"].append(extra)
        result = prepare_timeline(self.report)
        bar = next(bar for bar in result["bars"] if bar.span_id == "unmatched-rank")
        self.assertEqual(bar.start_us + bar.translation_us, 8)
        self.assertTrue(
            any("shared profile origin" in note for note in result["warnings"])
        )

    def test_module_envelope_keeps_internal_gaps(self):
        span = self.report["matches"][0]["profiled"]
        span["details"]["service_intervals"] = [[2000, 2001], [2003, 2005]]
        bar = next(
            bar
            for bar in prepare_timeline(self.report, detail="module")["bars"]
            if bar.side == "profiled"
        )
        self.assertEqual(bar.end_us - bar.start_us, 5)
        self.assertEqual(sum(b - a for a, b in bar.intervals), 3)

    def test_overlapping_activities_are_not_hidden(self):
        added = copy.deepcopy(self.report["matches"][0])
        for side in ("simulated", "profiled"):
            added[side]["id"] += ":overlap"
        self.report["matches"].append(added)
        rows = pack_tracks(prepare_timeline(self.report)["bars"], "profiled")
        self.assertEqual(len(rows), 2)
        added["profiled"]["details"]["device_activities"][0]["stream"] = 8
        rows = pack_tracks(prepare_timeline(self.report)["bars"], "profiled")
        self.assertEqual(len(rows), 2)
        self.assertTrue(any("stream 8" in name for name, _ in rows))

    def test_adjacent_endpoint_roundoff_does_not_create_false_overlap(self):
        first, second = [pair["simulated"] for pair in self.report["matches"]]
        second["start_us"] = math.nextafter(first["end_us"], 0)
        rows = pack_tracks(prepare_timeline(self.report)["bars"], "simulated")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1][1].start_us, second["start_us"])
        second["start_us"] = first["end_us"] - 0.1
        rows = pack_tracks(prepare_timeline(self.report)["bars"], "simulated")
        self.assertEqual(len(rows), 2)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "optional plot dependency"
    )
    def test_png_svg_cli_and_geometry_manifest(self):
        output = self.root / "images/layer0"
        result = export_timeline(
            self.report_path,
            output,
            forward=0,
            layer=0,
            alignment="selection",
            dpi=72,
            width=10,
        )
        self.assertEqual(len(result["events"]), 4)
        self.assertEqual(
            Path(str(output) + ".png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
        )
        root = ET.parse(str(output) + ".svg").getroot()
        text = " ".join(root.itertext())
        self.assertIn("Generated trace with TokenSim", text)
        self.assertIn("gemm_kernel", text)
        self.assertIn("stream 7", text)
        sidecar = json.loads(Path(str(output) + ".timeline.json").read_text())
        self.assertEqual(sidecar["comparison_sha256"], result["comparison_sha256"])
        for event in sidecar["events"]:
            self.assertEqual(
                event["end_us"] - event["start_us"],
                event["display_end_us"] - event["display_start_us"],
            )
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "TokenSim.profiler.plot",
                "--comparison",
                str(self.report_path),
                "--output",
                str(self.root / "cli"),
                "--formats",
                "svg",
                "--rank",
                "0",
                "--detail",
                "module",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["events"], 4)


if __name__ == "__main__":
    unittest.main()
