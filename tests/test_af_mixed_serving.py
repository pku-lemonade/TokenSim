"""Mixed serving and recomputation retain AF computation and communication."""

from unittest.mock import patch
import unittest

import simpy

from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import SimulationStateError
from TokenSim.llm.engine.af_worker import run_worker_compute
from TokenSim.llm.llm_engine import LLMEngine, LLMWorker
from TokenSim.llm.llm_request import Request
from tests.test_af_executor import (
    _cluster_dict,
    af_test_hardware,
    FixedComputeProvider,
    FixedTransferProvider,
)
from tests.test_af_operator_provider import _moe


class MixedServingTest(unittest.TestCase):
    def engine(self, *, capacity=None, compute=None, moe=False):
        cluster = ClusterConfig(**_cluster_dict(af_block_size=8))
        cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
        hardware = af_test_hardware(capacity_gib=0.1, moe=_moe() if moe else None)
        env = simpy.Environment()
        compute = compute or FixedComputeProvider(0.001, 0.002)
        engine = LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=4, batching="paged-attn", max_num_seqs=2
            ),
            af_compute_provider=compute,
            af_transfer_provider=FixedTransferProvider(0.001, 0.001),
        )
        return env, engine, compute

    def test_staggered_moe_requests_never_use_the_complete_backend(self):
        env, engine, _ = self.engine(moe=True)
        requests = [Request(1, 8, 4, 4), Request(2, 8, 3, 4)]
        batches = []
        original = LLMWorker.step

        def step(worker, running, latency):
            batches.append(tuple(req.is_prefill for req in running))
            return original(worker, running, latency)

        def source():
            requests[0].arrive(env)
            engine.add_requests([requests[0]])
            yield env.timeout(0.001)
            requests[1].arrive(env)
            engine.add_requests([requests[1]])

        with (
            patch.object(
                LLMWorker,
                "dynamic_batch",
                side_effect=AssertionError("complete backend called"),
            ),
            patch.object(LLMWorker, "step", step),
        ):
            env.process(source())
            env.run()
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        self.assertTrue(any(len(set(batch)) == 2 for batch in batches))
        self.assertEqual(len(engine.timing_recorder.timing.time), 2)

    def test_memory_pressure_recomputes_through_the_af_graph(self):
        env, engine, compute = self.engine(capacity=16)
        requests = [Request(i, 8, 8, 4) for i in (1, 2)]
        with patch.object(
            LLMWorker,
            "dynamic_batch",
            side_effect=AssertionError("complete backend called"),
        ):
            for req in requests:
                req.arrive(env)
                engine.add_requests_burst([req])
            env.run()
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        self.assertGreater(sum(req.recomputed_tokens_total for req in requests), 0)
        self.assertTrue(
            any(
                q.phase == "prefill" and q.context_lengths[0] > 8
                for q in compute.queries
            )
        )
        self.assertEqual(engine.af_executor.active_iterations, {})

    def test_failed_group_cancels_other_inflight_groups_and_releases_resources(self):
        class FailPrefill(FixedComputeProvider):
            def estimate(self, query):
                if query.phase == "prefill":
                    raise ValueError("prefill failed")
                return super().estimate(query)

        env, engine, _ = self.engine(compute=FailPrefill(0.001, 0.002))
        decode, prefill = Request(1, 8, 4, 4), Request(2, 8, 4, 4)
        decode.generation_idx = 1
        process = env.process(
            run_worker_compute(engine.request_workers[0], [decode, prefill])
        )
        with self.assertRaisesRegex(SimulationStateError, "prefill failed"):
            env.run(until=process)
        self.assertEqual(engine.af_executor.active_iterations, {})
        for resource in engine.af_resources.compute_slots.values():
            self.assertEqual(resource.count, 0)
            self.assertFalse(resource.queue)
