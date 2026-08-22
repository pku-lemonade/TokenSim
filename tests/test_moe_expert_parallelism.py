from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.config.config import CacheConfig, ClusterConfig, ParallelConfig, ParallelRankInfo, WorkerGroupConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.errors import ConfigurationError, WorkloadValidationError
from TokenSim.latency import RooflineLatencyBackend
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.moe import ExpertRouting, build_expert_placement, normalize_expert_histogram
from benchmark import build_parallel_config, validate_moe_parallel_config
from util.request import LLMSource
from util.results import export_result, get_moe_stats
from TokenSim.workload.loaders import load_json_pairs_workload


class _MoERoofline:
    def __init__(self) -> None:
        self.hardwares = {
            "TestGPU": SimpleNamespace(
                Name="TestGPU",
                MM_Card_Num=1,
                MM_TFLOPS=100,
                Capacity=1,
                Nvlink="nvlink-test",
            )
        }
        self.models = {
            "MoEModel": SimpleNamespace(
                Name="MoEModel",
                Nhead=8,
                Dmodel=128,
                Nlayer=4,
                Max_Token=4096,
                Multi_Query=False,
                Grouped_Query=False,
                FFN_Hidden=256,
            )
        }
        self.links = {
            "nvlink-test": SimpleNamespace(Latency=1e-6, UniBW=100.0),
            "ethernet-test": SimpleNamespace(Latency=2e-5, UniBW=10.0),
        }

    def Compute_Timebreakdown_Iteration(
        self,
        prefill_len,
        generation_idx,
        batch_size,
        model,
        hardware,
        Pipeline_Stage,
    ):
        return 0.001, 0.0002


def _moe_psla(parallel_config: ParallelConfig | None = None) -> PSLAConfig:
    return PSLAConfig(
        name="moe-test",
        model="MoEModel",
        distribution="burst",
        prefill_mean_len=16,
        prefill_range_len=0,
        decode_mean_len=2,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
        parallel_config=parallel_config,
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        num_shared_experts=1,
        num_moe_layers=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
    )


class MoEConfigTest(unittest.TestCase):
    def test_dense_defaults_and_moe_config_parse(self):
        dense = PSLAConfig.from_file("./data/psla/llama-7b.json")
        moe = PSLAConfig.from_file("./data/psla/moe-toy.json")

        self.assertFalse(dense.moe_config.enabled)
        self.assertTrue(moe.moe_config.enabled)
        self.assertEqual(moe.moe_config.num_experts, 4)
        self.assertEqual(moe.moe_config.num_experts_per_tok, 2)
        self.assertEqual(moe.moe_config.hidden_size, 128)

    def test_invalid_moe_config_and_ep_on_dense_fail(self):
        with self.assertRaises(ConfigurationError):
            PSLAConfig(
                name="bad",
                model="MoEModel",
                distribution="burst",
                prefill_mean_len=1,
                prefill_range_len=0,
                decode_mean_len=1,
                decode_range_len=0,
                decode_len_distribution="uniform",
                first_token_latency=MetricData(0, 0, 0),
                decode_token_latency=MetricData(0, 0, 0),
                qps=1,
                is_moe_model=True,
                num_experts=2,
                num_experts_per_tok=3,
                num_moe_layers=1,
            )
        dense = PSLAConfig.from_file("./data/psla/llama-7b.json")
        with self.assertRaises(ConfigurationError):
            validate_moe_parallel_config(
                dense,
                ParallelConfig(enable_expert_parallel=True),
            )

    def test_parallel_config_surface(self):
        config = ParallelConfig(
            enable_expert_parallel=True,
            expert_placement_strategy="round_robin",
            all2all_backend="deepep_low_latency",
        )

        self.assertEqual(
            set(k for k in config.to_dict() if "expert" in k or "all2all" in k),
            {"enable_expert_parallel", "expert_placement_strategy", "all2all_backend"},
        )


class MoEPlacementRoutingTest(unittest.TestCase):
    def test_linear_and_round_robin_placement(self):
        moe = _moe_psla().moe_config
        linear = build_expert_placement(
            moe,
            ParallelConfig(
                tensor_parallel_size=2,
                data_parallel_size=2,
                enable_expert_parallel=True,
            ),
            total_layers=4,
        )
        rr = build_expert_placement(
            moe,
            ParallelConfig(
                tensor_parallel_size=2,
                data_parallel_size=2,
                enable_expert_parallel=True,
                expert_placement_strategy="round_robin",
            ),
            total_layers=4,
        )

        self.assertEqual(linear.rank_to_experts, {0: [0], 1: [1], 2: [2], 3: [3]})
        self.assertEqual(rr.rank_to_experts, {0: [0], 1: [1], 2: [2], 3: [3]})
        self.assertEqual(linear.pp_stage_to_moe_layers[0], [1, 2])

    def test_routing_histogram_deterministic_and_validated(self):
        routing = ExpertRouting(_moe_psla().moe_config, seed=1)
        req = Request(0, prefill_len=4, decode_len=1, block_size=16)

        self.assertEqual(routing.histogram_for_step([req]), {0: 2, 1: 2, 2: 2, 3: 2})
        self.assertEqual(
            normalize_expert_histogram({"0": 3, "2": 1}, num_experts=4),
            {0: 3, 2: 1},
        )
        with self.assertRaises(WorkloadValidationError):
            normalize_expert_histogram({4: 1}, num_experts=4, request_id=7)

    def test_json_pairs_loader_uses_and_validates_expert_histogram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "moe_pairs.json"
            path.write_text(
                json.dumps([[8, 2, {"prefill_expert_histogram": {"0": 3, "1": 1}}]])
            )

            workload = load_json_pairs_workload(
                str(path),
                request_count=1,
                num_experts=4,
            )

        self.assertEqual(workload[0].prefill_expert_histogram, {0: 3, 1: 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_moe_pairs.json"
            path.write_text(json.dumps([[8, 2, {"expert_histogram": {"9": 1}}]]))
            with self.assertRaises(WorkloadValidationError):
                load_json_pairs_workload(str(path), request_count=1, num_experts=4)

    def test_skew_routing_has_higher_imbalance_than_uniform(self):
        moe_config = _moe_psla().moe_config
        uniform = ExpertRouting(moe_config)
        moe_config.routing.distribution = "skew"
        moe_config.routing.hot_experts = [0]
        moe_config.routing.hot_expert_fraction = 0.75
        skew = ExpertRouting(moe_config)

        uniform_hist = uniform.synthetic_histogram(16)
        skew_hist = skew.synthetic_histogram(16)

        self.assertGreater(max(skew_hist.values()), max(uniform_hist.values()))


class MoEMemoryLatencyIntegrationTest(unittest.TestCase):
    def test_expert_weights_are_sharded_and_kv_bytes_preserved(self):
        roofline = _MoERoofline()
        config = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
        moe = _moe_psla().moe_config
        placement = build_expert_placement(moe, config, total_layers=4)
        rank0 = ParallelRankInfo.from_global_rank(0, config)
        rank1 = ParallelRankInfo.from_global_rank(1, config)

        cache0 = CacheConfig(16, "TestGPU", "MoEModel", roofline, config, rank0, moe, placement)
        cache1 = CacheConfig(16, "TestGPU", "MoEModel", roofline, config, rank1, moe, placement)

        self.assertEqual(cache0.size_per_token, cache1.size_per_token)
        self.assertEqual(cache0.local_kv_heads, cache1.local_kv_heads)
        self.assertLess(cache0.model_param_size, cache0.model_param_size_unsharded)

    def test_latency_records_moe_compute_all2all_and_imbalance(self):
        roofline = _MoERoofline()
        config = ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
        )
        moe = _moe_psla().moe_config
        placement = build_expert_placement(moe, config, total_layers=4)
        rank = ParallelRankInfo.from_global_rank(0, config)
        backend = RooflineLatencyBackend(
            roofline,
            "MoEModel",
            "TestGPU",
            config,
            rank,
            moe_config=moe,
            expert_placement=placement,
        )
        req = Request(
            0,
            prefill_len=16,
            decode_len=1,
            block_size=16,
            prefill_expert_histogram={0: 20, 1: 2, 2: 2, 3: 2},
        )

        latency = backend.estimate_step_latency([req])
        stats = backend.moe_stats.as_dict()

        self.assertGreater(latency, 0)
        self.assertGreater(stats["moe_compute_latency"], 0)
        self.assertGreater(stats["moe_all2all_latency"], 0)
        self.assertGreater(stats["moe_expert_load_imbalance_ratio"], 1)

    def test_engine_runs_and_exports_moe_metrics(self):
        reset_g_time()
        env = simpy.Environment()
        roofline = _MoERoofline()
        config = ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
        )
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
        psla = _moe_psla()
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=psla,
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
            raise AssertionError("moe request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        stats = get_moe_stats(engine)

        self.assertTrue(stats["effective_moe_config"]["is_moe_model"])
        self.assertGreater(stats["moe_compute_latency"], 0)
        self.assertGreater(stats["moe_all2all_latency"], 0)
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
                model_config=psla,
                cluster=cluster,
                request_count=1,
                prefill_lens=[16],
                decode_lens=[2],
                requests=requests,
                notdone=[],
                duration=monitor.value,
                simulator_wall_time=0.123,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())
        self.assertGreater(result["moe_compute_latency"], 0)
        self.assertGreater(result["parallel_ep_all2all_latency"], 0)
        self.assertEqual(result["simulator_wall_time"], 0.123)
        self.assertEqual(result["simulated_time"], monitor.value)


if __name__ == "__main__":
    unittest.main()
