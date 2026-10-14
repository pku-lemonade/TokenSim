"""Scheduler query ranges are preserved through AF microbatch construction."""

import unittest

from TokenSim.af.builder import AFBlockBuilder
from TokenSim.af.planning import ParallelExecutionPlanner
from TokenSim.llm.llm_request import Request
from tests.test_af_executor import _cluster_dict
from TokenSim.config.cluster_config import ClusterConfig


class QueryGeometryTest(unittest.TestCase):
    def test_initial_chunk_does_not_read_uncomputed_prompt_keys(self):
        req = Request(1, 64, 3, 4)
        req.scheduled_tokens = 8
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        block = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(block.context_lengths, (8,))
        self.assertEqual(block.query_lengths, (8,))
        self.assertEqual(block.query_offsets, (0,))
        self.assertEqual(block.logits_rows, 0)
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(block),
            tuple((1, i) for i in range(8)),
        )

    def test_cached_suffix_uses_actual_query_positions_and_full_kv(self):
        req = Request(1, 64, 3, 4)
        req.start_context_build(56)
        req.scheduled_tokens = 8
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        block = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(block.context_lengths, (64,))
        self.assertEqual(block.query_offsets, (56,))
        self.assertEqual(block.logits_rows, 1)
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(block),
            tuple((1, i) for i in range(56, 64)),
        )
