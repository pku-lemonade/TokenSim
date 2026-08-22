from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.config.config import (
    CacheConfig,
    ClusterConfig,
    ParallelConfig,
    ParallelRankInfo,
    WorkerGroupConfig,
    local_kv_heads,
)
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.errors import ConfigurationError
from TokenSim.latency import RooflineLatencyBackend
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.parallel import ParallelCommunicator
from TokenSim.placement import DataParallelWorkerPool, RoundRobinWorkerPool
from benchmark import build_parallel_config
from util.request import LLMSource
from util.results import export_result, get_parallel_stats


class _ParallelRoofline:
    def __init__(self) -> None:
        self.hardwares = {
            "TestGPU": SimpleNamespace(
                MM_Card_Num=1,
                Capacity=0.001,
                Nvlink="nvlink-test",
            )
        }
        self.models = {
            "TestModel": SimpleNamespace(
                Name="TestModel",
                Nhead=8,
                Dmodel=16,
                Nlayer=4,
                Multi_Query=False,
                Grouped_Query=False,
            )
        }
        self.links = {
            "nvlink-test": SimpleNamespace(Latency=1e-6, UniBW=100.0),
            "ethernet-test": SimpleNamespace(Latency=2e-5, UniBW=10.0),
        }
        self.calls = []

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
        return 0.001, 0.0002


def _psla(parallel_config: ParallelConfig | None = None) -> PSLAConfig:
    return PSLAConfig(
        name="parallel-test",
        model="TestModel",
        distribution="burst",
        prefill_mean_len=16,
        prefill_range_len=0,
        decode_mean_len=1,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
        parallel_config=parallel_config,
    )


class _PlacementWorker:
    def __init__(self, worker_id: int, dp_rank: int) -> None:
        self.id = worker_id
        self.dp_rank = dp_rank


class ParallelConfigTest(unittest.TestCase):
    def test_cli_overrides_cluster_and_model_parallel_config(self):
        args = argparse.Namespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=None,
            data_parallel_size=2,
            data_parallel_rank=None,
            data_parallel_size_local=None,
        )
        cluster = ClusterConfig(
            num_workers=8,
            networks={"net1": "ethernet-test"},
            parallel_config=ParallelConfig(tensor_parallel_size=2),
        )
        model = _psla(ParallelConfig(pipeline_parallel_size=3))

        config = build_parallel_config(args, cluster, model)

        self.assertEqual(config.tensor_parallel_size, 4)
        self.assertEqual(config.pipeline_parallel_size, 1)
        self.assertEqual(config.data_parallel_size, 2)

    def test_worker_rank_mapping_and_count_validation(self):
        config = ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
        )
        cluster = ClusterConfig(
            num_workers=8,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="TestGPU",
                    num_workers=8,
                    network="net1",
                )
            ],
        )

        workers = cluster.workers(config)

        self.assertEqual(workers[0].rank_info, ParallelRankInfo(0, 0, 0, 0, 0, "dp0-pp0"))
        self.assertEqual(workers[3].rank_info, ParallelRankInfo(3, 3, 1, 1, 0, "dp0-pp1"))
        self.assertEqual(workers[4].rank_info, ParallelRankInfo(4, 0, 0, 0, 1, "dp1-pp0"))
        with self.assertRaises(ConfigurationError):
            cluster.workers(ParallelConfig(tensor_parallel_size=3))


class ParallelMemoryTest(unittest.TestCase):
    def test_cache_config_shards_weight_layers_and_kv_heads(self):
        roofline = _ParallelRoofline()
        config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        rank = ParallelRankInfo.from_global_rank(3, config)

        cache = CacheConfig(16, "TestGPU", "TestModel", roofline, config, rank)

        self.assertEqual(cache.num_layers_per_rank, 2)
        self.assertEqual(cache.local_kv_heads, 4)
        self.assertEqual(cache.size_per_token, 4 * 2 * 2 * 2 * 2)
        self.assertEqual(cache.model_param_size, cache.model_param_size_unsharded / 4)

    def test_unsupported_kv_head_divisibility_fails(self):
        with self.assertRaises(ConfigurationError):
            local_kv_heads(3, 2)


class ParallelLatencyTest(unittest.TestCase):
    def test_roofline_uses_parallel_config_and_records_sync_events(self):
        roofline = _ParallelRoofline()
        config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        rank = ParallelRankInfo.from_global_rank(0, config)
        workers = [
            SimpleNamespace(
                id=0,
                network="net1",
                nettype="ethernet-test",
                hardware="TestGPU",
                dp_rank=0,
                tp_rank=0,
                pp_rank=0,
            ),
            SimpleNamespace(
                id=1,
                network="net2",
                nettype="ethernet-test",
                hardware="TestGPU",
                dp_rank=0,
                tp_rank=1,
                pp_rank=0,
            ),
            SimpleNamespace(
                id=2,
                network="net2",
                nettype="ethernet-test",
                hardware="TestGPU",
                dp_rank=0,
                tp_rank=0,
                pp_rank=1,
            ),
        ]
        communicator = ParallelCommunicator(
            roofline=roofline,
            workers=workers,
            worker_id=0,
            rank_info=rank,
            parallel_config=config,
            hardware="TestGPU",
        )
        backend = RooflineLatencyBackend(
            roofline,
            "TestModel",
            "TestGPU",
            config,
            rank,
            communicator,
        )
        request = Request(0, prefill_len=16, decode_len=1, block_size=16)

        backend.estimate_step_latency([request])

        self.assertTrue(all(call[-1] == 2 for call in roofline.calls))
        self.assertTrue(
            all("tokensim_parallel_tp2_pp2" in call[-2] for call in roofline.calls)
        )
        self.assertEqual(
            roofline.hardwares["TestGPU__tokensim_parallel_tp2_pp2"].MM_Card_Num,
            2,
        )
        self.assertEqual(communicator.stats.roofline_conversion_count, 1)
        self.assertEqual(communicator.stats.sync_event_count, 3)
        self.assertGreater(communicator.stats.tp_collective_latency, 0)
        self.assertGreater(communicator.stats.pp_transfer_latency, 0)
        self.assertEqual(
            communicator.stats.link_type_counts,
            {"network": communicator.stats.sync_event_count},
        )

    def test_tensor_parallel_converts_projection_and_attention_compute(self):
        baseline_roofline = _ParallelRoofline()
        tp_roofline = _ParallelRoofline()
        request = Request(0, prefill_len=16, decode_len=1, block_size=16)

        baseline = RooflineLatencyBackend(
            baseline_roofline,
            "TestModel",
            "TestGPU",
            ParallelConfig(),
        )
        tp_backend = RooflineLatencyBackend(
            tp_roofline,
            "TestModel",
            "TestGPU",
            ParallelConfig(tensor_parallel_size=2),
        )

        baseline_latency = baseline.estimate_step_latency([request])
        tp_latency = tp_backend.estimate_step_latency([request])

        self.assertLess(tp_latency, baseline_latency)


class DataParallelPlacementTest(unittest.TestCase):
    def test_dp_pool_spreads_prefill_and_keeps_decode_in_group(self):
        workers = [_PlacementWorker(0, 0), _PlacementWorker(1, 1)]
        pool = DataParallelWorkerPool(workers, RoundRobinWorkerPool)
        req0 = SimpleNamespace()
        req1 = SimpleNamespace()

        first = pool.select_prefill_worker(req0)
        second = pool.select_prefill_worker(req1)
        decode = pool.select_transfer_target([req0])

        self.assertEqual(first.dp_rank, 0)
        self.assertEqual(second.dp_rank, 1)
        self.assertEqual(decode.dp_rank, req0.dp_rank)
        self.assertEqual(pool.dp_placement_counts, {0: 2, 1: 1})


class ParallelIntegrationTest(unittest.TestCase):
    def test_parallel_engine_runs_and_exports_metrics(self):
        reset_g_time()
        env = simpy.Environment()
        roofline = _ParallelRoofline()
        config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        cluster = ClusterConfig(
            num_workers=4,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="TestGPU",
                    num_workers=4,
                    network="net1",
                )
            ],
        )
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=_psla(),
            cluster_config=cluster,
            parallel_config=config,
            roofline=roofline,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
        requests = [Request(0, 16, 2, block_size=16)]
        env.process(LLMSource(env, engine, requests, qps=1, distribution="burst"))

        def stop_when_done():
            while env.now < 1:
                if requests[0].is_done:
                    duration = env.now
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return duration
                yield env.timeout(1e-6)
            raise AssertionError("parallel request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        stats = get_parallel_stats(engine)

        self.assertEqual(stats["parallel_expected_rank_count"], 4)
        self.assertEqual(stats["parallel_actual_rank_count"], 4)
        self.assertGreaterEqual(stats["parallel_sync_event_count"], 3)
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
                model_config=_psla(),
                cluster=cluster,
                request_count=1,
                prefill_lens=[16],
                decode_lens=[2],
                requests=requests,
                notdone=[],
                duration=monitor.value,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())
        self.assertEqual(result["parallel_config"]["tensor_parallel_size"], 2)
        self.assertEqual(result["parallel_config"]["pipeline_parallel_size"], 2)
        self.assertGreaterEqual(result["parallel_sync_event_count"], 3)


if __name__ == "__main__":
    unittest.main()
