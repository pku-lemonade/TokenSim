from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig, _GB
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.latency.base import DECODE_SCALE, PREFILL_OFFSET_SECONDS, PREFILL_SCALE
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, RequestStatus, g_time, reset_g_time
from util.request import LLMSource
from util.results import export_result, get_connector_stats


class _DeterministicRoofline:
    def __init__(self) -> None:
        self.hardwares = {
            "TestGPU": SimpleNamespace(
                MM_Card_Num=1,
                Capacity=0.0002,
                Nvlink="nvlink-test",
            )
        }
        self.models = {
            "TestModel": SimpleNamespace(
                Dmodel=1,
                Nlayer=1,
            )
        }
        self.links = {
            "nvlink-test": SimpleNamespace(Latency=1e-5, UniBW=100.0),
            "ethernet-test": SimpleNamespace(Latency=2e-5, UniBW=10.0),
        }
        self.calls: list[tuple[int, int, int, str, str, int]] = []

    def Compute_Timebreakdown_Iteration(
        self,
        prefill_len,
        generation_idx,
        batch_size,
        model,
        hardware,
        Pipeline_Stage,
    ):
        self.calls.append(
            (prefill_len, generation_idx, batch_size, model, hardware, Pipeline_Stage)
        )
        if generation_idx == 0:
            return 0.001, 0.0002
        return 0.002, 0.0003


class _LightweightCacheConfig:
    def __init__(self, block_size, hardware, model, roofline) -> None:
        self.block_size = block_size
        self.model = model
        model_config = roofline.models[model]
        self.size_per_token = model_config.Dmodel * 2 * 2 * model_config.Nlayer
        self.num_gpu_blocks = 64
        self.num_cpu_blocks = 64


def _psla_config() -> PSLAConfig:
    return PSLAConfig(
        name="test",
        model="TestModel",
        distribution="burst",
        prefill_mean_len=32,
        prefill_range_len=0,
        decode_mean_len=2,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
    )


def _pd_cluster() -> ClusterConfig:
    return ClusterConfig(
        num_workers=2,
        networks={"net1": "ethernet-test"},
        worker_groups=[
            WorkerGroupConfig(
                role="prefill",
                hardware="TestGPU",
                num_workers=1,
                network="net1",
            ),
            WorkerGroupConfig(
                role="decode",
                hardware="TestGPU",
                num_workers=1,
                network="net1",
            ),
        ],
        kv_transfer=KVTransferConfig(kv_connector="P2PConnector", kv_parallel_size=2),
    )


def _run_single_request_p2p() -> tuple[LLMEngine, Request, _DeterministicRoofline, float]:
    reset_g_time()
    env = simpy.Environment()
    roofline = _DeterministicRoofline()
    cluster = _pd_cluster()
    with patch("TokenSim.llm.llm_engine.CacheConfig", _LightweightCacheConfig):
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=_psla_config(),
            cluster_config=cluster,
            roofline=roofline,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
    request = Request(id=0, prefill_len=32, decode_len=2, block_size=16)

    env.process(
        LLMSource(
            env=env,
            engine=engine,
            requests=[request],
            qps=1,
            distribution="burst",
        )
    )

    def stop_when_done():
        deadline = 0.1
        while env.now < deadline:
            if request.is_done:
                duration = env.now
                engine.send_task(engine, Task.STOP)
                yield env.timeout(1e-6)
                return duration
            yield env.timeout(1e-6)
        raise AssertionError("P2P integration request did not finish")

    monitor = env.process(stop_when_done())
    env.run(until=monitor)
    return engine, request, roofline, monitor.value


class KVTransferIntegrationTest(unittest.TestCase):
    def test_p2p_transfer_path_has_precise_metrics_and_request_timing(self):
        engine, request, roofline, duration = _run_single_request_p2p()

        model = roofline.models["TestModel"]
        size_per_token = model.Dmodel * 2 * 2 * model.Nlayer
        expected_blocks = 2
        expected_bytes = expected_blocks * 16 * size_per_token
        expected_transfer_latency = (
            1e-5 + expected_bytes / _GB / 100.0
        )
        expected_prefill_latency = (0.001 + 0.0002) * PREFILL_SCALE + (
            PREFILL_OFFSET_SECONDS
        )
        expected_decode_latency = (0.002 + 0.0003) * DECODE_SCALE

        stats = get_connector_stats(engine)

        self.assertTrue(request.is_done)
        self.assertEqual(request.status, RequestStatus.FINISHED_STOPPED)
        self.assertEqual(len(g_time.time), 1)
        self.assertEqual(g_time.time[0].id, request.id)
        self.assertEqual(stats["connector_transfer_count"], 2)
        self.assertEqual(stats["connector_transfer_blocks"], expected_blocks * 2)
        self.assertEqual(stats["connector_transfer_bytes"], expected_bytes * 2)
        self.assertTrue(
            math.isclose(
                stats["connector_transfer_latency"],
                expected_transfer_latency * 2,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                stats["connector_load_wait_time"],
                expected_transfer_latency,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                stats["connector_save_wait_time"],
                expected_transfer_latency,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        self.assertEqual(stats["connector_producer_count"], 1)
        self.assertEqual(stats["connector_consumer_count"], 1)
        self.assertEqual(stats["connector_placement_decision_count"], 2)
        self.assertEqual(request.prefill_batch_size, 1)
        self.assertEqual(request.decode_batch_sum, 1)
        self.assertTrue(
            math.isclose(
                request.prefill_service_time,
                expected_prefill_latency,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                request.decode_service_time_sum,
                expected_decode_latency,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            duration > request.prefill_service_time + request.decode_service_time_sum
        )
        self.assertTrue(
            math.isclose(
                request.decode_time_sum,
                expected_transfer_latency + expected_decode_latency,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )

    def test_p2p_result_export_contains_precise_connector_metrics(self):
        engine, request, _, duration = _run_single_request_p2p()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                qps=1,
                batching="paged-attn",
                results_path=tmpdir,
                cluster="unused.json",
            )
            export_result(
                args=args,
                g_time=g_time,
                engine=engine,
                model_config=_psla_config(),
                cluster=_pd_cluster(),
                request_count=1,
                prefill_lens=[request.prefill_len],
                decode_lens=[request.decode_len],
                requests=[request],
                notdone=[],
                duration=duration,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())

        stats = get_connector_stats(engine)
        for key, value in stats.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["connector_transfer_count"], 2)
        self.assertEqual(result["connector_transfer_blocks"], 4)
        self.assertEqual(result["connector_transfer_bytes"], 256)


if __name__ == "__main__":
    unittest.main()
