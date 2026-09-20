from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.comm.collectives import CollectiveModel
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
from TokenSim.hardware.topology import TopologyPlacement
from TokenSim.latency import OperatorTableLatencyBackend
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.parallel import ParallelCommunicator
from TokenSim.placement import DataParallelWorkerPool, RoundRobinWorkerPool
from benchmark import build_parallel_config
from tests.hardware_fixtures import (
    test_device,
    test_hardware,
    test_links,
    test_model,
    test_topology,
)
from util.request import LLMSource
from util.results import export_result, get_parallel_stats

_DEVICE = test_device("TestGPU", memory_gib=0.05)
_MODEL = test_model("TestModel", hidden_size=16, intermediate_size=32, num_layers=4, num_attention_heads=8)


def _hardware():
    return test_hardware(_DEVICE, models=[_MODEL])


def _placement(node_size: int = 8, mapping: dict[int, int] | None = None) -> TopologyPlacement:
    return TopologyPlacement(test_topology(node_size), mapping or {})


def _communicator(workers, worker_id, rank, config, placement=None):
    from TokenSim.hardware.links import LinkCatalog

    placement = placement or _placement()
    model = CollectiveModel(placement.topology, LinkCatalog(test_links()), family="nvidia_gpu")
    return ParallelCommunicator(
        placement=placement,
        collective_model=model,
        workers=workers,
        worker_id=worker_id,
        rank_info=rank,
        parallel_config=config,
    )


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
    def test_total_kv_capacity_is_split_across_dp_replicas(self):
        config = ParallelConfig(data_parallel_size=8)
        cluster = ClusterConfig(
            num_workers=8,
            networks={"net1": "ethernet-test"},
            kv_cache_capacity_tokens_total=21_943_624,
        )

        self.assertEqual(
            cluster.effective_kv_cache_capacity_per_dp_rank(config),
            2_742_953,
        )

    def test_total_and_per_dp_kv_capacity_are_mutually_exclusive(self):
        with self.assertRaises(ConfigurationError):
            ClusterConfig(
                num_workers=1,
                networks={"net1": "ethernet-test"},
                kv_cache_capacity_tokens_total=1024,
                kv_cache_capacity_tokens_per_dp_rank=128,
            )

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

        self.assertEqual(
            workers[0].rank_info, ParallelRankInfo(0, 0, 0, 0, 0, "dp0-pp0")
        )
        self.assertEqual(
            workers[3].rank_info, ParallelRankInfo(3, 3, 1, 1, 0, "dp0-pp1")
        )
        self.assertEqual(
            workers[4].rank_info, ParallelRankInfo(4, 0, 0, 0, 1, "dp1-pp0")
        )
        with self.assertRaises(ConfigurationError):
            cluster.workers(ParallelConfig(tensor_parallel_size=3))


class ParallelMemoryTest(unittest.TestCase):
    def test_cache_config_shards_weight_layers_and_kv_heads(self):
        config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        rank = ParallelRankInfo.from_global_rank(3, config)

        cache = CacheConfig(16, _DEVICE, _MODEL, config, rank)

        self.assertEqual(cache.num_layers_per_rank, 2)
        self.assertEqual(cache.local_kv_heads, 4)
        # 4 local heads x head_dim 2 x (K,V) x fp16 x 2 local layers
        self.assertEqual(cache.size_per_token, 4 * 2 * 2 * 2 * 2)
        self.assertAlmostEqual(cache.model_param_size, cache.model_param_size_unsharded / 4)

    def test_unsupported_kv_head_divisibility_fails(self):
        with self.assertRaises(ConfigurationError):
            local_kv_heads(3, 2)

    def test_reserved_memory_reduces_kv_blocks(self):
        from TokenSim.hardware.device import DeviceSpec

        plain = CacheConfig(16, _DEVICE, _MODEL)
        reserved_device = DeviceSpec.simple(
            "ReservedGPU",
            family="nvidia_gpu",
            peak_flops=100e12,
            memory_capacity_bytes=_DEVICE.memory_capacity_bytes.value,
            memory_bandwidth_bytes_per_s=1e12,
            memory_reserved_bytes=_DEVICE.memory_capacity_bytes.value / 2,
        )
        reserved = CacheConfig(16, reserved_device, _MODEL)

        self.assertLess(reserved.num_gpu_blocks, plain.num_gpu_blocks)
        self.assertAlmostEqual(reserved.usable_memory_bytes, plain.usable_memory_bytes / 2)
        self.assertEqual(reserved.to_dict()["reserved_bytes"], reserved_device.memory_reserved_bytes.value)

    def test_kv_heads_replicate_when_fewer_than_tp(self):
        self.assertEqual(local_kv_heads(1, 4), 1)
        self.assertEqual(local_kv_heads(8, 4), 2)


class ParallelLatencyTest(unittest.TestCase):
    def test_backend_records_tp_and_pp_sync_events_across_topology_levels(self):
        config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        rank = ParallelRankInfo.from_global_rank(0, config)
        workers = [
            SimpleNamespace(id=0, dp_rank=0, tp_rank=0, pp_rank=0),
            SimpleNamespace(id=1, dp_rank=0, tp_rank=1, pp_rank=0),
            SimpleNamespace(id=2, dp_rank=0, tp_rank=0, pp_rank=1),
        ]
        # worker 0 and 1 share a node; worker 2 sits on another node
        placement = _placement(node_size=2, mapping={0: 0, 1: 1, 2: 2})
        communicator = _communicator(workers, 0, rank, config, placement)
        backend = OperatorTableLatencyBackend(
            device=_DEVICE,
            model=_MODEL,
            parallel_config=config,
            rank_info=rank,
            communicator=communicator,
            fallback="analytical_only",
        )
        request = Request(0, prefill_len=16, decode_len=1, block_size=16)

        latency = backend.estimate_step_latency([request])

        self.assertGreater(latency, 0)
        self.assertEqual(backend.layers_local, 2)
        self.assertEqual(communicator.stats.tp_shard_event_count, 1)
        self.assertEqual(communicator.stats.sync_event_count, 2)
        self.assertGreater(communicator.stats.tp_collective_latency, 0)
        self.assertGreater(communicator.stats.pp_transfer_latency, 0)
        self.assertEqual(
            communicator.stats.link_type_counts,
            {"node": 1, "cluster": 1},
        )
        self.assertEqual(backend.describe()["fallback"], "analytical_only")
        self.assertEqual(backend.stats.match_type_counts["analytical"], sum(backend.stats.match_type_counts.values()))

    def test_tensor_parallel_shards_compute(self):
        request = Request(0, prefill_len=16, decode_len=1, block_size=16)

        baseline = OperatorTableLatencyBackend(
            device=_DEVICE, model=_MODEL, parallel_config=ParallelConfig(), fallback="analytical_only"
        )
        tp_backend = OperatorTableLatencyBackend(
            device=_DEVICE,
            model=_MODEL,
            parallel_config=ParallelConfig(tensor_parallel_size=2),
            fallback="analytical_only",
        )

        baseline_latency = baseline.estimate_step_latency([request])
        tp_latency = tp_backend.estimate_step_latency([request])

        # Without a communicator no all-reduce cost is added, so the TP shard
        # must be strictly cheaper than the unsharded model.
        self.assertLess(tp_latency, baseline_latency)
        self.assertEqual(tp_backend.heads_local, 4)
        self.assertEqual(tp_backend.inter_local, 16)

    def test_ep_all2all_uses_moe_shape_and_records_event(self):
        config = ParallelConfig(tensor_parallel_size=1, data_parallel_size=4, enable_expert_parallel=True, all2all_backend="deepep_low_latency")
        rank = ParallelRankInfo.from_global_rank(0, config)
        workers = [SimpleNamespace(id=i, dp_rank=i, tp_rank=0, pp_rank=0) for i in range(4)]
        communicator = _communicator(workers, 0, rank, config, _placement(node_size=4))

        latency = communicator.estimate_ep_all2all(0, count=2, num_tokens=16, hidden_size=4096, top_k=2, num_experts=8)
        plain = communicator.estimate_ep_all2all(16 * 2 * 4096 * 2, count=1)

        self.assertGreater(latency, 0)
        self.assertEqual(communicator.stats.sync_event_count, 2)
        self.assertGreater(communicator.stats.ep_all2all_latency, 0)
        # deepep_low_latency scales the analytical all-to-all by 0.5 per phase, two phases per layer
        self.assertAlmostEqual(latency, 2 * 2 * 0.5 * plain)

    def test_expert_parallel_group_follows_scope_and_stays_on_its_node(self):
        # 64 ranks laid out as 8 nodes x 8 GPUs, TP8 inside a node, DP8 across nodes.
        workers = [SimpleNamespace(id=i, dp_rank=i // 8, tp_rank=i % 8, pp_rank=0) for i in range(64)]
        placement = _placement(node_size=8)
        layouts = {}
        for scope, size in (("per_dp", 8), ("global", None)):
            config = ParallelConfig(
                tensor_parallel_size=8,
                data_parallel_size=8,
                enable_expert_parallel=True,
                expert_parallel_scope=scope,
                expert_parallel_size=size,
                all2all_backend="deepep_high_throughput",
            )
            rank = ParallelRankInfo.from_global_rank(21, config)  # dp2, tp5
            communicator = _communicator(workers, 21, rank, config, placement)
            communicator.estimate_ep_all2all(0, count=1, num_tokens=64, hidden_size=7168, top_k=8, num_experts=256)
            layouts[scope] = communicator.describe()
            self.assertEqual(communicator.tp_group(), tuple(range(16, 24)))

        per_dp, wide = layouts["per_dp"], layouts["global"]
        self.assertEqual(per_dp["ep_group"], list(range(16, 24)))
        self.assertEqual((per_dp["ep_group_size"], per_dp["ep_group_count"]), (8, 8))
        self.assertEqual(per_dp["ep_group_fan"], [8, 1])
        self.assertEqual(per_dp["ep_group_link"], "node")
        self.assertEqual((wide["ep_group_size"], wide["ep_group_count"]), (64, 1))
        self.assertEqual(wide["ep_group_fan"], [8, 8])
        self.assertEqual(wide["ep_group_link"], "cluster")

    def test_collective_cost_grows_with_group_size_and_topology_span(self):
        from TokenSim.comm.collectives import CollectiveQuery
        from TokenSim.hardware.links import LinkCatalog

        topology = test_topology(node_size=4)
        model = CollectiveModel(topology, LinkCatalog(test_links()), family="nvidia_gpu")
        message = 1 << 20
        intra = model.estimate(CollectiveQuery("all_reduce", message, topology.group_layout([0, 1, 2, 3])))
        inter = model.estimate(CollectiveQuery("all_reduce", message, topology.group_layout([0, 1, 2, 3, 4, 5, 6, 7])))
        pair = model.estimate(CollectiveQuery("all_reduce", message, topology.group_layout([0, 1])))

        self.assertLess(pair.latency_us, intra.latency_us)
        self.assertLess(intra.latency_us, inter.latency_us)
        self.assertEqual([level.level for level in inter.levels], ["node", "cluster"])
        self.assertEqual(inter.match_type, "analytical")


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
        hardware = _hardware()
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
            hardware=hardware,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
            latency_backend_type="analytical",
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
        self.assertEqual(stats["parallel_groups"]["tp_group"], [0, 1])
        self.assertEqual(stats["parallel_groups"]["ep_group_size"], 1)
        self.assertGreaterEqual(stats["parallel_sync_event_count"], 2)
        self.assertGreater(stats["parallel_tp_collective_latency"], 0)
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
        self.assertEqual(result["parallel_groups"]["tp_group_link"], "node")
        self.assertGreaterEqual(result["parallel_sync_event_count"], 2)
        self.assertEqual(result["latency_backends"]["TestGPU"]["backend"], "operator_table")
        self.assertGreater(result["operator_query_count"], 0)


if __name__ == "__main__":
    unittest.main()
