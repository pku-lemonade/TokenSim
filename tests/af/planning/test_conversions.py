"""Conversion launches cover batched slices without duplicating network bytes."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    execution_plan,
    af_test_resources,
)
from tests.af.support.model import _moe


class ConversionBatchTest(unittest.TestCase):
    def run_batch(self, *, f_ranks=1, coalesce=False):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=f_ranks, f_dp=1, f_ep=f_ranks)
        cluster = replace(
            cluster,
            af=replace(
                cluster.af,
                af_block_size=8,
                activation_dtype="bf16",
                coalesce_ready_transfers=coalesce,
                links=[
                    replace(cluster.af.links[0], a_to_f_dtype="fp8", f_to_a_dtype="fp8")
                ],
            ),
        )
        env = simpy.Environment()
        compute, transfer = FixedComputeProvider(1e-6, 1e-6), FixedTransferProvider(
            0, 0
        )
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            compute,
            transfer,
            "TestModel",
            "bf16",
            1,
            16,
            runtime_plan=execution_plan(cluster, moe=_moe()),
            moe_config=_moe(),
            resources=af_test_resources(env, cluster),
        )
        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
            )
            for i in range(8)
        ]
        iteration = executor.submit(0, requests)
        env.process(executor.run_iteration(iteration))
        env.run()
        return executor, compute, transfer

    def test_entire_block_converts_once_in_each_direction(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                executor, compute, transfer = self.run_batch(coalesce=coalesce)
                for stage in ("encode", "decode"):
                    conversions = [q for q in compute.queries if q.stage == stage]
                    self.assertEqual(len(conversions), 2)
                    self.assertEqual(
                        sorted(q.local_shape["elements"] for q in conversions),
                        [8 * 16, 8 * 2 * 16],
                    )
                self.assertEqual(
                    sum(
                        q.payload_bytes
                        for q in transfer.queries
                        if q.operation == "af_a_to_f"
                    ),
                    8 * 16,
                )
                self.assertEqual(
                    sum(
                        q.payload_bytes
                        for q in transfer.queries
                        if q.operation == "af_f_to_a"
                    ),
                    8 * 2 * 16,
                )
                executor.trace.validate_invariants(0)

    def test_quantization_is_shared_across_expert_destinations(self):
        executor, compute, _ = self.run_batch(f_ranks=2)
        encoded = [
            q
            for q in compute.queries
            if q.stage == "encode" and q.physical_worker == "a-pool:0"
        ]
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded[0].local_shape["elements"], 8 * 16)
        by_id = {e.node_id: e for e in executor.trace.events}
        for event in executor.trace.events:
            if event.operation == "af_a_to_f":
                for parent in event.predecessor_ids:
                    self.assertEqual(by_id[parent].stage, "encode")
