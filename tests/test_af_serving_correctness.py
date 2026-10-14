"""Serving configurations preserve AF execution and resource ownership."""

import unittest

import simpy

from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from tests.test_af_executor import (
    af_test_hardware,
    FixedComputeProvider,
    FixedTransferProvider,
)
from tests.test_af_parallel_planning import _cluster


class AFServingCorrectnessTest(unittest.TestCase):
    def make_engine(self, *, utilization=1.0, capacity=None):
        hardware = af_test_hardware(capacity_gib=0.1)
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
        return LLMEngine(
            env=simpy.Environment(),
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=4, batching="paged-attn", gpu_memory_utilization=utilization
            ),
            af_compute_provider=FixedComputeProvider(0.001, 0.001),
            af_transfer_provider=FixedTransferProvider(0.001, 0.001),
        )

    def test_memory_utilization_limits_every_physical_budget(self):
        full = self.make_engine()
        half = self.make_engine(utilization=0.5)
        self.assertLess(
            half.request_workers[0].cache_config.num_gpu_blocks,
            full.request_workers[0].cache_config.num_gpu_blocks // 2,
        )
        for original, limited in zip(
            full.af_runtime_plan.physical_states, half.af_runtime_plan.physical_states
        ):
            self.assertEqual(limited.capacity_bytes, original.capacity_bytes // 2)

    def test_explicit_kv_limit_caps_all_attention_shards(self):
        engine = self.make_engine(capacity=32)
        manager = engine.request_workers[0].scheduler.block_manager
        self.assertEqual(manager.num_total_gpu_blocks, 8)
        for rank in manager.rank_keys:
            self.assertEqual(manager.rank_status(rank)["gpu_capacity"], 8)
        with self.assertRaisesRegex(ConfigurationError, "cannot fit"):
            engine.validate_request_capacity([Request(1, 64, 2, 4)])

    def test_kv_limit_cannot_manufacture_physical_memory(self):
        full = self.make_engine()
        capped = self.make_engine(capacity=10**12)
        self.assertEqual(
            full.request_workers[0].cache_config.num_gpu_blocks,
            capped.request_workers[0].cache_config.num_gpu_blocks,
        )

    def test_lockstep_is_rejected_before_starting_workers(self):
        for ranks in (1, 2):
            with self.subTest(ranks=ranks):
                env = simpy.Environment()
                hardware = af_test_hardware(capacity_gib=0.1)
                with self.assertRaisesRegex(ConfigurationError, "dp_lockstep.*AF"):
                    LLMEngine(
                        env=env,
                        kv_transfer_config=KVTransferConfig.default(),
                        model_spec=hardware.model("TestModel"),
                        cluster_config=_cluster(
                            a_tp=1, a_dp=ranks, f_tp=1, f_dp=1, f_ep=1
                        ),
                        hardware=hardware,
                        parallel_config=ParallelConfig(data_parallel_size=ranks),
                        config=SimulationConfig(dp_lockstep=True),
                    )
                self.assertEqual(len(env._queue), 0)
