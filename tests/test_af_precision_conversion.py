"""Wire precision conversion is priced and participates in the AF DAG."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.hardware.device import DeviceCatalog
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.workload import elementwise_work
from tests.test_af_executor import af_test_hardware
from tests.test_af_parallel_planning import _cluster


class PrecisionConversionTest(unittest.TestCase):
    def test_cast_accounts_for_source_reads_and_target_writes(self):
        for source, target, byte_width in (("bf16", "fp8", 3), ("fp8", "fp32", 5)):
            work = elementwise_work(f"cast_{source}_to_{target}", 2, 16, target)
            self.assertEqual(work.activation_bytes, 32 * byte_width)
            self.assertGreater(work.flops, 0)
        with self.assertRaisesRegex(ValueError, "target dtype"):
            elementwise_work("cast_bf16_to_fp8", 1, 16, "bf16")

    def test_builtin_provider_executes_wire_codecs_with_tp_shards(self):
        for tp in (1, 2):
            with self.subTest(tp=tp):
                cluster = _cluster(a_tp=tp, a_dp=1, f_tp=tp, f_dp=1, f_ep=1)
                cluster = replace(
                    cluster,
                    af=replace(
                        cluster.af,
                        latency_provider="operator_table",
                        activation_dtype="bf16",
                        links=[
                            replace(
                                cluster.af.links[0],
                                a_to_f_dtype="fp8",
                                f_to_a_dtype="fp32",
                            )
                        ],
                    ),
                )
                hardware = af_test_hardware(capacity_gib=0.1)
                hardware = replace(
                    hardware,
                    devices=DeviceCatalog(
                        [
                            replace(
                                device,
                                peak_compute={
                                    **device.peak_compute,
                                    "fp32": device.peak_compute["fp16"],
                                },
                            )
                            for device in hardware.devices
                        ]
                    ),
                )
                env = simpy.Environment()
                engine = LLMEngine(
                    env=env,
                    kv_transfer_config=KVTransferConfig.default(),
                    model_spec=hardware.model("TestModel"),
                    cluster_config=cluster,
                    hardware=hardware,
                    config=SimulationConfig(
                        block_size=4, latency_fallback="analytical_only"
                    ),
                )
                req = Request(1, 8, 4, 4)
                req.generation_idx = 1
                executor = engine.af_executor
                iteration = executor.submit(0, [req], "decode")
                env.run(until=env.process(executor.run_iteration(iteration)))
                codecs = [
                    e for e in executor.trace.events if e.stage in {"encode", "decode"}
                ]
                self.assertTrue(codecs)
                self.assertTrue(all(e.service_duration > 0 for e in codecs))
                operations = {
                    e.details["components"][0]["key"]["op_name"] for e in codecs
                }
                self.assertEqual(
                    operations,
                    {
                        "cast_bf16_to_fp8",
                        "cast_fp8_to_bf16",
                        "cast_bf16_to_fp32",
                        "cast_fp32_to_bf16",
                    },
                )
                executor.trace.validate_invariants(iteration.iteration_id)
