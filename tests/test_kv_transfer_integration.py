from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.hardware.links import LinkCatalog
from TokenSim.latency.base import LatencyBackend
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, RequestStatus, g_time, reset_g_time
from tests.hardware_fixtures import test_device, test_hardware, test_links, test_model
from util.request import LLMSource
from util.results import export_result, get_connector_stats

PREFILL_LATENCY = 0.0012
DECODE_LATENCY = 0.0023

# Tiny model so that a 32-token request spans exactly two 16-token blocks and
# the KV bytes are easy to reason about: 1 head x head_dim 1 x (K,V) x fp16 x 1 layer = 4 B/token.
_MODEL = test_model("TestModel", hidden_size=1, intermediate_size=2, num_layers=1, num_attention_heads=1, vocab_size=4)
_DEVICE = test_device("TestGPU", memory_gib=0.0002)


class _DeterministicBackend(LatencyBackend):
    """Fixed service times so timing assertions stay exact."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def estimate_step_latency(self, requests):
        if requests[0].is_prefill or getattr(requests[0], "needs_recompute", False):
            self.calls.append(("prefill", len(requests)))
            return PREFILL_LATENCY
        self.calls.append(("decode", len(requests)))
        return DECODE_LATENCY


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
            WorkerGroupConfig(role="prefill", hardware="TestGPU", num_workers=1, network="net1"),
            WorkerGroupConfig(role="decode", hardware="TestGPU", num_workers=1, network="net1"),
        ],
        kv_transfer=KVTransferConfig(kv_connector="P2PConnector", kv_parallel_size=2),
    )


def _expected_transfer_latency(bytes_: int) -> float:
    # Both workers share network net1, so the synthesized topology puts them in
    # one node connected by the device's scale-up link.
    link = LinkCatalog(test_links()).get("nvlink-test")
    return link.transfer_us(bytes_) * 1e-6


def _run_single_request_p2p() -> tuple[LLMEngine, Request, _DeterministicBackend, float]:
    reset_g_time()
    env = simpy.Environment()
    cluster = _pd_cluster()
    backend = _DeterministicBackend()
    with patch("TokenSim.llm.llm_engine.build_latency_backend", return_value=backend):
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=_psla_config(),
            cluster_config=cluster,
            hardware=test_hardware(_DEVICE, models=[_MODEL]),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
    request = Request(id=0, prefill_len=32, decode_len=2, block_size=16)

    env.process(LLMSource(env=env, engine=engine, requests=[request], qps=1, distribution="burst"))

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
    return engine, request, backend, monitor.value


class KVTransferIntegrationTest(unittest.TestCase):
    def test_p2p_transfer_path_has_precise_metrics_and_request_timing(self):
        engine, request, backend, duration = _run_single_request_p2p()

        size_per_token = engine.workers[0].cache_config.size_per_token
        self.assertEqual(size_per_token, 4)
        expected_blocks = 2
        expected_bytes = expected_blocks * 16 * size_per_token
        expected_transfer_latency = _expected_transfer_latency(expected_bytes)

        stats = get_connector_stats(engine)

        self.assertTrue(request.is_done)
        self.assertEqual(request.status, RequestStatus.FINISHED_STOPPED)
        self.assertEqual(len(g_time.time), 1)
        self.assertEqual(g_time.time[0].id, request.id)
        self.assertEqual(backend.calls, [("prefill", 1), ("decode", 1)])
        self.assertEqual(stats["connector_transfer_count"], 2)
        self.assertEqual(stats["connector_transfer_blocks"], expected_blocks * 2)
        self.assertEqual(stats["connector_transfer_bytes"], expected_bytes * 2)
        self.assertTrue(
            math.isclose(stats["connector_transfer_latency"], expected_transfer_latency * 2, rel_tol=0, abs_tol=1e-15)
        )
        self.assertTrue(
            math.isclose(stats["connector_load_wait_time"], expected_transfer_latency, rel_tol=0, abs_tol=1e-15)
        )
        self.assertTrue(
            math.isclose(stats["connector_save_wait_time"], expected_transfer_latency, rel_tol=0, abs_tol=1e-15)
        )
        self.assertEqual(stats["connector_producer_count"], 1)
        self.assertEqual(stats["connector_consumer_count"], 1)
        self.assertEqual(stats["connector_placement_decision_count"], 2)
        self.assertEqual(request.prefill_batch_size, 1)
        self.assertEqual(request.decode_batch_sum, 1)
        self.assertTrue(math.isclose(request.prefill_service_time, PREFILL_LATENCY, rel_tol=0, abs_tol=1e-15))
        self.assertTrue(math.isclose(request.decode_service_time_sum, DECODE_LATENCY, rel_tol=0, abs_tol=1e-15))
        self.assertTrue(duration > request.prefill_service_time + request.decode_service_time_sum)
        self.assertTrue(
            math.isclose(
                request.decode_time_sum,
                expected_transfer_latency + DECODE_LATENCY,
                rel_tol=0,
                abs_tol=1e-15,
            )
        )

    def test_p2p_result_export_contains_precise_connector_metrics(self):
        engine, request, _, duration = _run_single_request_p2p()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(qps=1, batching="paged-attn", results_path=tmpdir, cluster="unused.json")
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
