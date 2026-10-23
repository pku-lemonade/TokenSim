"""Exact semantic transfer and temporary instrumentation cleanup."""

from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.collection import bind_captures, collection_inputs
from TokenSim.profiler.paired import kernel_correspondence
from TokenSim.profiler.importer import read_trace
from scripts.collect.operator_trace_worker import AnnotationSession
from scripts.collect.collect_operator_trace import run_generation
from tests.trace.test_profiler import fixtures


def paired_fixture(directory):
    root = Path(directory)
    _, structure = fixtures(root)
    with gzip.open(structure, "rt") as stream:
        document = json.load(stream)
    rows = document["traceEvents"] if isinstance(document, dict) else document
    kernels = [dict(row) for row in rows if row.get("cat") == "kernel"]
    for index, row in enumerate(kernels):
        row.update(ts=100000 + index * 20, dur=7 + index)
    timing = root / "timing" / "profile_tp0.json"
    timing.parent.mkdir()
    timing.write_text(json.dumps(kernels))
    manifest = dict(
        batch_size=1,
        input_tokens=1,
        output_tokens=1,
        paired_captures=bind_captures([structure], [timing], root),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return structure, timing, manifest


class PairedTraceTest(unittest.TestCase):
    def test_collection_selects_declared_timing_files_and_rejects_incomplete_or_escaping_inputs(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, timing, manifest = paired_fixture(directory)
            manifest.update(
                status="completed", trace_files=[str(timing.relative_to(root))]
            )
            path = root / "selected.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(collection_inputs(path), [timing])
            manifest["explicit_manifest_marker"] = "selected"
            path.write_text(json.dumps(manifest))
            selected = read_trace([timing], manifest_path=path)
            self.assertEqual(
                selected.inputs[0]["manifest"]["explicit_manifest_marker"], "selected"
            )
            for change in (
                dict(status="failed"),
                dict(trace_files=["../outside.json"]),
            ):
                path.write_text(json.dumps(dict(manifest, **change)))
                with self.assertRaises(ConfigurationError):
                    collection_inputs(path)

    def test_changed_launch_dimensions_do_not_inherit_a_semantic_scope(self):
        left = dict(
            ph="X",
            cat="kernel",
            name="gemm",
            ts=0,
            dur=1,
            _index=0,
            args=dict(device=0, stream=1, grid=[1, 1, 1]),
        )
        right = dict(left, args=dict(device=0, stream=1, grid=[2, 1, 1]))
        with self.assertRaisesRegex(ConfigurationError, "grid differs"):
            kernel_correspondence([left], [right])

    def test_an_extra_forward_cannot_be_labeled_as_another_decode_step(self):
        with tempfile.TemporaryDirectory() as directory:
            structure, timing, _ = paired_fixture(directory)
            with self.assertRaisesRegex(ConfigurationError, "forward count"):
                bind_captures(
                    [structure], [timing], Path(directory), expected_forwards=2
                )

    def test_batch_admission_holds_all_requests_until_the_burst_is_enqueued(self):
        calls = []

        def enqueue(*args, **kwargs):
            calls.append("enqueue")

        client = SimpleNamespace(call_utility=lambda *args: calls.append(args[0]))
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(engine_core=client),
            enqueue=enqueue,
            wait_for_completion=lambda **kwargs: calls.append("wait"),
        )
        run_generation(llm, ["a", "b"], None)
        self.assertEqual(
            calls, ["pause_scheduler", "enqueue", "resume_scheduler", "wait"]
        )
        calls.clear()

        def fail(*args, **kwargs):
            raise RuntimeError("enqueue failed")

        llm.enqueue = fail
        with self.assertRaises(RuntimeError):
            run_generation(llm, ["a", "b"], None)
        self.assertEqual(calls, ["pause_scheduler", "resume_scheduler"])

    def test_ownership_comes_from_structure_and_times_come_from_gpu_only_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            structure, timing, _ = paired_fixture(directory)
            spans = read_trace(timing).spans
            self.assertEqual([span.operation for span in spans], ["qkv", "output"])
            self.assertEqual([span.duration_us for span in spans], [7, 8])
            self.assertEqual([span.start_us for span in spans], [100000, 100020])
            self.assertTrue(
                all(
                    span.details["attribution"] == "paired_kernel_sequence"
                    for span in spans
                )
            )
            self.assertTrue(all(span.source == str(timing) for span in spans))

    def test_changed_structure_or_timing_bytes_fail_the_pinned_binding(self):
        for target in ("structure", "timing"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                structure, timing, _ = paired_fixture(directory)
                path = structure if target == "structure" else timing
                if path.suffix == ".gz":
                    with path.open("ab") as stream:
                        stream.write(b"changed")
                else:
                    path.write_text(path.read_text() + " ")
                with self.assertRaises(ConfigurationError):
                    read_trace(timing)

    def test_different_kernel_identity_or_count_cannot_borrow_module_ownership(self):
        for changed in ("identity", "count"):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                structure, timing, _ = paired_fixture(directory)
                rows = json.loads(timing.read_text())
                if changed == "identity":
                    rows[0]["name"] = "another_kernel"
                else:
                    rows.pop()
                timing.write_text(json.dumps(rows))
                with self.assertRaises(ConfigurationError):
                    bind_captures([structure], [timing], Path(directory))

    def test_structure_reference_cannot_escape_the_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            _, timing, manifest = paired_fixture(directory)
            manifest["paired_captures"][0]["structure_trace"] = "../outside.json"
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ConfigurationError, "escapes"):
                read_trace(timing)

    def test_method_instrumentation_restores_the_inherited_method(self):
        class Model:
            def compute_logits(self, value):
                if value < 0:
                    raise ValueError("failed")
                return value + 1

        model = Model()
        torch = SimpleNamespace(
            profiler=SimpleNamespace(record_function=lambda label: nullcontext())
        )
        with patch.dict("sys.modules", {"torch": torch}):
            session = AnnotationSession()
            session.method(model, "compute_logits", "tokensim.model.lm_head")
            self.assertEqual(model.compute_logits(3), 4)
            with self.assertRaises(ValueError):
                model.compute_logits(-1)
            session.close()
            session.close()
        self.assertNotIn("compute_logits", vars(model))
        self.assertEqual(model.compute_logits(3), 4)
