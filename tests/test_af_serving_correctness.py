"""Serving configurations preserve AF execution and resource ownership."""

import unittest

import simpy

from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMEngine
from tests.test_af_executor import af_test_hardware
from tests.test_af_parallel_planning import _cluster


class AFServingCorrectnessTest(unittest.TestCase):
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
