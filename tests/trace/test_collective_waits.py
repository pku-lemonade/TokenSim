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
    def test_arrival_wait_preserves_residence_and_shared_clock_alignment(self):
        pairs, observed = experiment()
        before = [(right.start_us, right.end_us, right.busy_us) for _, right in pairs]
        observed.diagnostics["collective_waits"] = analyze_collectives(pairs, observed)
        self.assertEqual(observed.diagnostics["collective_waits"]["analyzed_calls"], 2)
        first = service_metrics(*pairs[0])
        self.assertEqual(first["profiled_collective_residence_us"], 16)
        self.assertEqual(first["profiled_peer_wait_us"], 10)
        self.assertEqual(first["profiled_device_service_us"], 6)
        self.assertEqual(service_metrics(*pairs[1])["profiled_peer_wait_us"], 0)
        self.assertEqual(
            before,
            [(right.start_us, right.end_us, right.busy_us) for _, right in pairs],
        )
        simulated = ImportedTrace([left for left, _ in pairs], {}, [], {})
        with tempfile.TemporaryDirectory() as directory:
            report = write_comparison(
                simulated, observed, pairs, [], [], Path(directory), mapping_hash="hash"
            )
        self.assertEqual(
            [a["translation_us"] for a in report["alignment"]["anchors"].values()],
            [-100, -100],
        )
        self.assertEqual(report["matches"][1]["start_offset_us"], -10)

    def test_compute_overlapping_a_wait_is_preserved_in_device_work(self):
        pairs, observed = experiment()
        right = pairs[0][1]
        right.details["service_intervals"].append((103, 108))
        right.details["compute_intervals"] = [(103, 108)]
        analyze_collectives(pairs, observed)
        self.assertEqual(service_metrics(*pairs[0])["profiled_device_service_us"], 11)

    def test_unqualified_collective_waits_remain_unknown(self):
        for case in ("clock", "peer", "barrier", "disjoint", "namespace"):
            with self.subTest(case=case):
                pairs, observed = experiment(proof=case != "clock")
                if case == "peer":
                    pairs.pop()
                elif case == "barrier":
                    pairs[0][1].details["collective_calls"][0][
                        "wait_model"
                    ] = "unresolved"
                elif case == "disjoint":
                    pairs[0][1].details["collective_calls"][0]["end_us"] = 105
                elif case == "namespace":
                    observed.inputs = deepcopy(observed.inputs)
                    observed.inputs[1]["manifest"]["worker_identity"][1][
                        "time_namespace"
                    ] = "another"
                result = analyze_collectives(pairs, observed)
                self.assertEqual(result["analyzed_calls"], 0)
                self.assertIsNone(
                    service_metrics(*pairs[0])["profiled_device_service_us"]
                )
                self.assertEqual(pairs[0][1].busy_us, 16)
