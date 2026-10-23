"""Peer arrival accounting requires complete groups and a proven clock basis."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from TokenSim.profiler.collectives import analyze_collectives, service_metrics
from TokenSim.profiler.importer import ImportedTrace
from TokenSim.profiler.records import Span
from TokenSim.profiler.reporting import write_comparison


def experiment(*, proof=True):
    identity = dict(hostname="host", boot_id="boot", time_namespace="time:[1]")
    manifest = (
        dict(
            measurement_protocol="paired_structure_gpu_timing_v1",
            worker_identity=[dict(identity, rank=rank) for rank in (0, 1)],
        )
        if proof
        else {}
    )
    inputs = [
        dict(
            path=f"rank{rank}",
            sha256="hash",
            manifest=manifest,
            metadata=dict(host_name="host", baseTimeNanoseconds=1000),
        )
        for rank in (0, 1)
    ]
    pairs = []
    for rank, start, end in ((0, 100, 116), (1, 110, 117)):
        left = Span(
            f"sim:{rank}",
            "all_reduce",
            str(rank),
            0,
            0,
            "decode",
            0,
            6,
            "collective",
            "sim",
            "simulation",
            busy_us=6,
            details=dict(participants=[0, 1]),
        )
        right = Span(
            f"gpu:{rank}",
            "all_reduce",
            str(rank),
            0,
            0,
            "decode",
            start,
            end,
            "collective",
            "gpu",
            f"clock:{rank}",
            busy_us=end - start,
            details=dict(
                service_intervals=[(start, end)],
                communication_intervals=[(start, end)],
                compute_intervals=[],
                collective_calls=[
                    dict(
                        operation="all_reduce",
                        start_us=start,
                        end_us=end,
                        wait_model="peer_start_barrier",
                    )
                ],
            ),
        )
        pairs.append((left, right))
    observed = ImportedTrace([right for _, right in pairs], {}, inputs, {})
    return pairs, observed


class CollectiveWaitTest(unittest.TestCase):
    def test_collective_without_a_proven_start_barrier_retains_unknown_service(self):
        pairs, observed = experiment()
        for _, right in pairs:
            right.details["collective_calls"][0]["wait_model"] = "unresolved"
        analysis = analyze_collectives(pairs, observed)
        self.assertEqual(analysis["analyzed_calls"], 0)
        self.assertIsNone(service_metrics(*pairs[0])["profiled_device_service_us"])

    def test_arrival_wait_is_separate_from_unchanged_kernel_residence(self):
        pairs, observed = experiment()
        before = [(right.start_us, right.end_us, right.busy_us) for _, right in pairs]
        result = analyze_collectives(pairs, observed)
        self.assertEqual(result["analyzed_calls"], 2)
        first = service_metrics(*pairs[0])
        self.assertEqual(first["profiled_collective_residence_us"], 16)
        self.assertEqual(first["profiled_peer_wait_us"], 10)
        self.assertEqual(first["profiled_device_service_us"], 6)
        self.assertEqual(first["device_service_error_us"], 0)
        self.assertEqual(service_metrics(*pairs[1])["profiled_peer_wait_us"], 0)
        self.assertEqual(
            before,
            [(right.start_us, right.end_us, right.busy_us) for _, right in pairs],
        )

    def test_unknown_clock_or_missing_participant_cannot_produce_a_service_estimate(
        self,
    ):
        for missing in (False, True):
            with self.subTest(missing=missing):
                pairs, observed = experiment(proof=missing)
                if missing:
                    pairs.pop()
                result = analyze_collectives(pairs, observed)
                self.assertEqual(result["analyzed_calls"], 0)
                self.assertIsNone(
                    service_metrics(*pairs[0])["profiled_device_service_us"]
                )
                self.assertEqual(pairs[0][1].busy_us, 16)

    def test_compute_overlapping_a_wait_is_preserved_in_device_work(self):
        pairs, observed = experiment()
        right = pairs[0][1]
        right.details["service_intervals"].append((103, 108))
        right.details["compute_intervals"] = [(103, 108)]
        analyze_collectives(pairs, observed)
        self.assertEqual(service_metrics(*pairs[0])["profiled_device_service_us"], 11)

    def test_nonoverlapping_collective_intervals_are_not_paired_by_duration(self):
        pairs, observed = experiment()
        pairs[0][1].details["collective_calls"][0]["end_us"] = 105
        result = analyze_collectives(pairs, observed)
        self.assertEqual(result["analyzed_calls"], 0)
        self.assertTrue(
            any(
                item["reason"] == "collective_intervals_do_not_overlap"
                for item in result["unresolved"]
            )
        )

    def test_shared_clock_alignment_keeps_rank_arrival_differences_visible(self):
        pairs, observed = experiment()
        observed.diagnostics["collective_waits"] = analyze_collectives(pairs, observed)
        simulated = ImportedTrace([left for left, _ in pairs], {}, [], {})
        with tempfile.TemporaryDirectory() as directory:
            report = write_comparison(
                simulated, observed, pairs, [], [], Path(directory), mapping_hash="hash"
            )
        self.assertTrue(report["alignment"]["cross_rank_overlap_available"])
        shifts = [
            anchor["translation_us"]
            for anchor in report["alignment"]["anchors"].values()
        ]
        self.assertEqual(shifts, [-100, -100])
        self.assertEqual(report["matches"][1]["start_offset_us"], -10)

    def test_host_or_time_namespace_mismatch_disables_cross_rank_attribution(self):
        pairs, observed = experiment()
        observed.inputs = deepcopy(observed.inputs)
        observed.inputs[1]["manifest"]["worker_identity"][1][
            "time_namespace"
        ] = "another"
        result = analyze_collectives(pairs, observed)
        self.assertFalse(result["shared_clock"]["available"])
        self.assertEqual(result["analyzed_calls"], 0)
