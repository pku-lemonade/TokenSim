import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import WorkloadValidationError
from scripts import analyze_lpx_gpu_trace as analyzer
from scripts.lpx.gpu_profile import file_record
from scripts.lpx.gpu_trace import summarize_decode
from tests.test_lpx_gpu_profile import output_fixture


def event(start, duration, name="kernel", category="kernel", device=0):
    return dict(cat=category, ph="X", ts=start, dur=duration, name=name, pid=device)


def selection(steps=3):
    return dict(
        basis="Synthetic decode boundaries; not measurements",
        decode_steps=[
            dict(start_us=10 + index * 10, end_us=20 + index * 10)
            for index in range(steps)
        ],
        categories=[],
    )


def category(name, indices):
    return dict(
        name=name, event_indices=indices, basis="Synthetic correlation evidence"
    )


def case_fixture(directory):
    work = Path(directory)
    control, row = output_fixture(work, "control")
    profiled, row = output_fixture(work, "profiled")
    trace = work / "trace.json.gz"
    with gzip.open(trace, "wt") as handle:
        json.dump(dict(traceEvents=[event(10, 30)]), handle)
    case = dict(
        status="collected",
        mode="graphs",
        context=10000,
        requests_sha256=row["config"]["requests_sha256"],
        environment_sha256=row["config"]["environment_sha256"],
        outputs=[file_record(control), file_record(profiled)],
        traces=[file_record(trace)],
        timing=dict(profiled_over_control=2),
    )
    chosen = dict(
        selection(), trace=str(trace), trace_sha256=file_record(trace)["sha256"]
    )
    case_path, selection_path = work / "case.json", work / "selection.json"
    case_path.write_text(json.dumps(case))
    selection_path.write_text(json.dumps(chosen))
    return case_path, selection_path, trace


class GpuTraceTest(unittest.TestCase):
    def test_partition_excludes_prefill_and_ignores_cpu_overlap(self):
        events = [
            event(0, 10, "prefill"),
            event(10, 6, "layer"),
            event(16, 2, "head"),
            event(18, 1, "sample"),
            event(21, 2, "unknown"),
            event(29, 3, "copy", "gpu_memcpy"),
            event(15, 20, "scheduler", "cpu_op"),
        ]
        chosen = selection()
        chosen["categories"] = [
            category("transformer_layers", [1]),
            category("vocabulary_head", [2]),
            category("logits_sampling", [3]),
        ]
        result = summarize_decode(events, chosen, 3)
        totals = {name: row["total_us"] for name, row in result["partitions"].items()}
        self.assertEqual(
            totals,
            dict(
                transformer_layers=6,
                vocabulary_head=2,
                logits_sampling=1,
                unclassified_gpu=5,
                gpu_category_overlap=0,
                no_gpu_activity=16,
            ),
        )
        self.assertEqual(result["gpu_busy_us"], 14)
        self.assertEqual(result["mean_step_us"], 10)
        self.assertEqual(result["selected_gpu_events"], 5)
        self.assertEqual(result["analysis_status"], "gpu_partition_only")
        self.assertAlmostEqual(
            sum(row["fraction"] for row in result["partitions"].values()), 1
        )
        for step in result["steps"]:
            self.assertEqual(sum(step["partitions_us"].values()), step["wall_us"])

    def test_same_category_concurrency_is_unioned(self):
        chosen = selection(1)
        chosen["categories"] = [category("transformer_layers", [0, 1])]
        result = summarize_decode([event(10, 10), event(12, 6)], chosen, 1)
        self.assertEqual(result["partitions"]["transformer_layers"]["total_us"], 10)
        self.assertEqual(result["activity_names"][0]["summed_activity_us"], 16)

    def test_distinct_category_overlap_is_not_double_counted(self):
        chosen = selection(2)
        chosen["categories"] = [
            category("transformer_layers", [0]),
            category("vocabulary_head", [1]),
        ]
        result = summarize_decode([event(10, 10), event(15, 10)], chosen, 2)
        for name in (
            "transformer_layers",
            "vocabulary_head",
            "gpu_category_overlap",
            "no_gpu_activity",
        ):
            self.assertEqual(result["partitions"][name]["total_us"], 5)

    def test_gpu_intervals_clip_to_decode_boundaries(self):
        result = summarize_decode([event(5, 10), event(19, 8)], selection(1), 1)
        self.assertEqual(result["gpu_busy_us"], 6)
        self.assertEqual(result["partitions"]["no_gpu_activity"]["total_us"], 4)

    def test_unknown_names_remain_unclassified(self):
        result = summarize_decode(
            [event(10, 10, "lm_head transformer sample")], selection(1), 1
        )
        self.assertEqual(result["partitions"]["unclassified_gpu"]["total_us"], 10)
        self.assertEqual(result["partitions"]["vocabulary_head"]["total_us"], 0)

    def test_window_count_and_evidence_are_required(self):
        for chosen, count in [
            (selection(), 2),
            (selection(), True),
            ({}, 3),
            (dict(selection(), basis=""), 3),
            ([], 3),
        ]:
            with (
                self.subTest(selection=chosen, count=count),
                self.assertRaises(WorkloadValidationError),
            ):
                summarize_decode([event(10, 30)], chosen, count)

    def test_windows_reject_gaps_overlap_and_invalid_numbers(self):
        for changed in (
            dict(start_us=21, end_us=30),
            dict(start_us=19, end_us=30),
            dict(start_us=20, end_us=20),
            dict(start_us=float("nan"), end_us=30),
            dict(start_us=20, end_us=float("inf")),
            dict(start_us=True, end_us=30),
            dict(start_us=-1, end_us=30),
            "bad",
        ):
            chosen = selection()
            chosen["decode_steps"][1] = changed
            with (
                self.subTest(window=changed),
                self.assertRaises(WorkloadValidationError),
            ):
                summarize_decode([event(10, 30)], chosen, 3)

    def test_assignments_reject_duplicates_bad_indices_and_cpu_events(self):
        for groups in (
            [category("unknown", [0])],
            [category("vocabulary_head", [0]), category("vocabulary_head", [1])],
            [category("vocabulary_head", [0]), category("logits_sampling", [0])],
            [category("vocabulary_head", [True])],
            [category("vocabulary_head", [-1])],
            [category("vocabulary_head", [2])],
            [category("vocabulary_head", [1])],
            [category("vocabulary_head", [])],
            [dict(category("vocabulary_head", [0]), basis="")],
        ):
            chosen = dict(selection(), categories=groups)
            with (
                self.subTest(groups=groups),
                self.assertRaises(WorkloadValidationError),
            ):
                summarize_decode(
                    [event(10, 30), event(10, 30, category="cpu_op")], chosen, 3
                )

    def test_assignments_cannot_refer_to_prefill_only(self):
        chosen = dict(selection(), categories=[category("transformer_layers", [0])])
        with self.assertRaises(WorkloadValidationError):
            summarize_decode([event(0, 5), event(10, 30)], chosen, 3)

    def test_gpu_events_require_valid_complete_single_device_intervals(self):
        invalid = [
            dict(event(10, 30), dur=-1),
            dict(event(10, 30), ts=float("nan")),
            dict(event(10, 30), ph="B"),
        ]
        missing = event(10, 30)
        del missing["pid"]
        invalid.append(missing)
        for broken in invalid:
            with self.subTest(event=broken), self.assertRaises(WorkloadValidationError):
                summarize_decode([broken], selection(), 3)
        with self.assertRaises(WorkloadValidationError):
            summarize_decode([event(10, 30), event(10, 30, device=1)], selection(), 3)
        with self.assertRaises(WorkloadValidationError):
            summarize_decode([event(10, 30, category="cpu_op")], selection(), 3)

    def test_analysis_verifies_case_and_preserves_pending_status(self):
        with tempfile.TemporaryDirectory() as directory:
            case_path, selection_path, trace = case_fixture(directory)
            result = analyzer.analyze(case_path, selection_path)
            self.assertEqual(result["expected_decode_steps"], 3)
            self.assertEqual(result["trace"], file_record(trace))
            self.assertEqual(result["analysis_status"], "gpu_partition_only")
            self.assertEqual(result["control_timing"]["profiled_over_control"], 2)
            summary = Path(directory) / "summary.json"
            self.assertEqual(
                analyzer.main(
                    [
                        "--case",
                        str(case_path),
                        "--selection",
                        str(selection_path),
                        "--summary",
                        str(summary),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(summary.read_text()), result)

    def test_analysis_rejects_changed_trace_or_selection_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            case_path, selection_path, trace = case_fixture(directory)
            original = selection_path.read_text()
            chosen = json.loads(original)
            chosen["trace_sha256"] = "changed"
            selection_path.write_text(json.dumps(chosen))
            with self.assertRaises(WorkloadValidationError):
                analyzer.analyze(case_path, selection_path)
            selection_path.write_text(original)
            trace.write_bytes(b"changed")
            with self.assertRaises(WorkloadValidationError):
                analyzer.analyze(case_path, selection_path)

    def test_analysis_rejects_changed_case_identity_or_token_count(self):
        with tempfile.TemporaryDirectory() as directory:
            case_path, selection_path, trace = case_fixture(directory)
            original = json.loads(case_path.read_text())
            for field, value in (
                ("requests_sha256", "changed"),
                ("status", "queued"),
                ("mode", "unknown"),
                ("context", 1),
            ):
                changed = copy.deepcopy(original)
                changed[field] = value
                case_path.write_text(json.dumps(changed))
                with (
                    self.subTest(field=field),
                    self.assertRaises(WorkloadValidationError),
                ):
                    analyzer.analyze(case_path, selection_path)
            case_path.write_text(json.dumps(original))
            chosen = json.loads(selection_path.read_text())
            chosen["decode_steps"].pop()
            selection_path.write_text(json.dumps(chosen))
            with self.assertRaises(WorkloadValidationError):
                analyzer.analyze(case_path, selection_path)

    def test_summary_cannot_overwrite_source_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            case_path, selection_path, trace = case_fixture(directory)
            for destination in (
                case_path,
                selection_path,
                trace,
                Path(directory) / "profiled.jsonl",
            ):
                before = destination.read_bytes()
                with (
                    self.subTest(destination=destination),
                    self.assertRaises(WorkloadValidationError),
                ):
                    analyzer.main(
                        [
                            "--case",
                            str(case_path),
                            "--selection",
                            str(selection_path),
                            "--summary",
                            str(destination),
                        ]
                    )
                self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
