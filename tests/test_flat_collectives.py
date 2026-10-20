"""Flat collective policies preserve per-rank bytes and explicit assumptions."""

from dataclasses import replace
import unittest

from TokenSim.comm.flat_collectives import flat_collective
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.links import LinkClass


class FlatCollectiveTest(unittest.TestCase):
    def setUp(self):
        self.link = LinkClass("flat", "test fabric", 100e9, 0.5)

    def estimate(self, operation="all_reduce", size=8192, algorithm="ring", ranks=8):
        return flat_collective(
            operation, ranks, size, self.link, algorithm=algorithm, dtype="bf16"
        )

    def test_ring_accounts_for_every_neighbor_round(self):
        value = self.estimate()
        self.assertEqual(value.detail["rounds"], 14)
        self.assertEqual(value.detail["wire_bytes"], 2 * 7 * 1024)
        self.assertEqual(value.latency_us, 14 * 0.5 + 14336 / 100e9 * 1e6)

    def test_auto_switches_from_startup_to_bandwidth_cost(self):
        small = self.estimate(algorithm="auto")
        large = self.estimate(size=8 << 20, algorithm="auto")
        self.assertEqual(small.detail["algorithm"], "tree")
        self.assertEqual(small.detail["rounds"], 6)
        self.assertEqual(large.detail["algorithm"], "ring")
        self.assertNotIn("direct", small.detail["algorithm_candidates"])

    def test_all_gather_uses_local_input_and_full_output(self):
        for algorithm, rounds in (("ring", 7), ("direct", 1)):
            with self.subTest(algorithm=algorithm):
                value = self.estimate("all_gather", 1024, algorithm)
                self.assertEqual(value.detail["input_bytes_per_rank"], 1024)
                self.assertEqual(value.detail["output_bytes_per_rank"], 8192)
                self.assertEqual(value.detail["wire_bytes"], 7168)
                self.assertEqual(value.detail["rounds"], rounds)
        self.assertEqual(
            self.estimate("reduce_scatter").detail["output_bytes_per_rank"], 1024
        )

    def test_empty_and_singleton_collectives_have_no_network_service(self):
        for algorithm in ("ring", "tree", "direct", "auto"):
            self.assertEqual(self.estimate(size=0, algorithm=algorithm).latency_us, 0)
            self.assertEqual(self.estimate(ranks=1, algorithm=algorithm).latency_us, 0)

    def test_link_efficiency_ramp_is_not_a_transport_startup_cost(self):
        first = self.estimate(algorithm="tree")
        self.link = replace(self.link, half_bandwidth_bytes=0, collective_efficiency=1)
        self.assertEqual(first, self.estimate(algorithm="tree"))

    def test_invalid_policy_and_fractional_dtype_shards_are_rejected(self):
        for algorithm in ("unknown", None, True):
            with self.assertRaises(ConfigurationError):
                self.estimate(algorithm=algorithm)
        with self.assertRaisesRegex(ConfigurationError, "integer number"):
            self.estimate(size=8194)
        with self.assertRaises(ConfigurationError):
            self.estimate(ranks=0)
