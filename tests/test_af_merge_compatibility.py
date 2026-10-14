"""Regression coverage for AF behavior across the main runtime merge."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.cli import build_parser, main
from TokenSim.comm.collectives import CollectiveModel
from TokenSim.errors import WorkloadValidationError
from TokenSim.hardware.links import LinkCatalog
from TokenSim.hardware.topology import TopologyLevel, TopologySpec
from TokenSim.llm.llm_request import Request
from TokenSim.moe import ExpertRouting, normalize_expert_histogram
from tests.support.af_config import _cluster_dict
from tests.support.af_hardware import af_test_hardware
from tests.support.af_providers import FixedComputeProvider, FixedTransferProvider
from tests.support.hardware import make_links, moe_test_model


class AFMergeCompatibilityTest(unittest.TestCase):
    def test_point_to_point_oversubscription_never_improves_latency(self):
        self.links = LinkCatalog(make_links())

        topology = TopologySpec(
            "three-level",
            (
                TopologyLevel("node", 4, "nvlink-test", links_per_endpoint=2),
                TopologyLevel("rack", 2, "ethernet-test"),
                TopologyLevel("cluster", None, "ethernet-test"),
            ),
        )
        for endpoints, level_index in (((0, 1), 0), ((0, 4), 1), ((0, 8), 2)):
            for payload in (0, 1, 1 << 20):
                with self.subTest(endpoints=endpoints, payload=payload):
                    layout = topology.group_layout(endpoints)
                    self.assertEqual(layout.lowest_common_level, level_index)
                    previous = -1
                    for oversubscription in (1, 2, 4):
                        levels = list(topology.levels)
                        levels[level_index] = replace(
                            levels[level_index], oversubscription=oversubscription
                        )
                        model = CollectiveModel(
                            replace(topology, levels=tuple(levels)), self.links
                        )
                        estimate = model.point_to_point(payload, layout)
                        self.assertGreaterEqual(estimate.latency_us, previous)
                        previous = estimate.latency_us
                        if payload:
                            self.assertEqual(
                                estimate.levels[0].level, levels[level_index].name
                            )

    def test_routing_validates_each_phase_and_does_not_reuse_prefill_cache(self):
        routing = ExpertRouting(moe_test_model().moe, seed=1)
        req = Request(
            0,
            prefill_len=4,
            decode_len=2,
            block_size=16,
            prefill_expert_histogram={0: 8},
            decode_expert_histogram={1: 2},
        )

        self.assertEqual(routing.histogram_for_step([req]), {0: 8})
        req.generation_idx = 1
        self.assertEqual(routing.histogram_for_step([req]), {1: 2})

        bad = Request(
            1,
            prefill_len=4,
            decode_len=1,
            block_size=16,
            prefill_expert_histogram={0: 7},
        )
        with self.assertRaisesRegex(WorkloadValidationError, "must equal 8"):
            routing.histogram_for_step([bad])

    def test_histogram_rejects_an_incorrect_route_count(self):
        with self.assertRaisesRegex(WorkloadValidationError, "must equal 8"):
            normalize_expert_histogram(
                {0: 3, 1: 4}, num_experts=4, request_id=7, expected_route_count=8
            )

    def test_current_cli_runs_af_and_exports_runtime_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cluster_path = path / "cluster.json"
            cluster_path.write_text(json.dumps(_cluster_dict()))
            args = build_parser().parse_args(
                [
                    "--cluster",
                    str(cluster_path),
                    "--model",
                    "TestModel",
                    "--latency_backend",
                    "analytical",
                    "--batching",
                    "paged-attn",
                    "--block_size",
                    "4",
                    "--qps",
                    "1",
                    "--distribution",
                    "burst",
                    "--request_count",
                    "2",
                    "--prefill_mean_len",
                    "8",
                    "--decode_mean_len",
                    "2",
                    "--prefill_range_len",
                    "0",
                    "--decode_range_len",
                    "0",
                    "--verbose",
                    "none",
                    "--results_path",
                    str(path / "results"),
                ]
            )
            args.af_compute_provider = FixedComputeProvider(0.001, 0.001)
            args.af_transfer_provider = FixedTransferProvider(0.001, 0.001)
            with (
                patch(
                    "TokenSim.cli.run.HardwareContext.load",
                    return_value=af_test_hardware(),
                ),
                redirect_stdout(io.StringIO()),
            ):
                engine = main(args)
            result = json.loads(
                next((path / "results").glob("result_*.json")).read_text()
            )
            self.assertTrue(result["af_enabled"])
            self.assertGreater(result["af_iteration_count"], 0)
            self.assertIsNotNone(result["af_runtime_state"])
            self.assertEqual(result["notdone"], [])
            self.assertEqual(len(engine.timing_recorder.timing.request_time), 2)
