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
from TokenSim.latency import OperatorTableLatencyBackend
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.moe import ExpertRouting, build_expert_placement, normalize_expert_histogram
from benchmark import build_parallel_config, validate_moe_parallel_config
from tests.hardware_fixtures import moe_test_model, test_device, test_hardware
from util.request import LLMSource
from util.results import export_result, get_moe_stats
from TokenSim.workload.loaders import load_json_pairs_workload

_DEVICE = test_device("TestGPU", memory_gib=1.0)
_MOE_MODEL = moe_test_model()


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
            {
                "enable_expert_parallel",
                "expert_parallel_scope",
                "expert_parallel_size",
                "expert_placement_strategy",
                "all2all_backend",
            },
        )

    def test_expert_parallel_size_is_validated_against_scope_and_layout(self):
        per_dp = ParallelConfig(
            tensor_parallel_size=8,
            data_parallel_size=8,
            enable_expert_parallel=True,
            expert_parallel_scope="per_dp",
            expert_parallel_size=8,
        )
        wide = ParallelConfig(tensor_parallel_size=8, data_parallel_size=8, enable_expert_parallel=True)
        paired = ParallelConfig(
            tensor_parallel_size=8, data_parallel_size=8, enable_expert_parallel=True, expert_parallel_size=16
        )

        # per_dp: 8 replicas of EP8 over the replica's own TP ranks
        self.assertEqual((per_dp.expert_parallel_group_size, per_dp.expert_parallel_group_count), (8, 8))
        self.assertEqual(per_dp.expert_parallel_replicas_per_group, 1)
        self.assertEqual(per_dp.expert_parallel_group(dp_rank=3, tp_rank=5), 3)
        self.assertEqual(per_dp.expert_parallel_rank(dp_rank=3, tp_rank=5), 5)
        # global (default): one wide-EP group over all 64 ranks fed by 8 replicas
        self.assertEqual((wide.expert_parallel_group_size, wide.expert_parallel_group_count), (64, 1))
        self.assertEqual(wide.expert_parallel_replicas_per_group, 8)
        self.assertEqual(wide.expert_parallel_rank(dp_rank=3, tp_rank=5), 29)
        # global with an explicit size: groups of whole replicas
        self.assertEqual((paired.expert_parallel_group_size, paired.expert_parallel_group_count), (16, 4))
        self.assertEqual(paired.expert_parallel_group(dp_rank=3, tp_rank=5), 1)
        # EP disabled: everything collapses to one group of size 1
        self.assertEqual(ParallelConfig(tensor_parallel_size=8).expert_parallel_group_size, 1)

        for bad in (
            {"expert_parallel_scope": "per_dp", "expert_parallel_size": 4},
            {"expert_parallel_size": 4},
            {"expert_parallel_size": 24},
            {"expert_parallel_size": 8, "enable_expert_parallel": False},
            {"expert_parallel_scope": "per_node"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ParallelConfig(
                    **{"tensor_parallel_size": 8, "data_parallel_size": 8, "enable_expert_parallel": True, **bad}
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
        self.assertEqual((linear.scope, linear.ep_rank_count, linear.ep_group_count), ("global", 4, 1))

    def test_per_dp_scope_gives_every_replica_a_full_copy_of_the_experts(self):
        moe = _moe_psla().moe_config
        config = ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
            expert_parallel_scope="per_dp",
            expert_parallel_size=2,
        )
        placement = build_expert_placement(moe, config, total_layers=4)

        self.assertEqual((placement.scope, placement.ep_rank_count, placement.ep_group_count), ("per_dp", 2, 2))
        self.assertEqual(placement.rank_to_experts, {0: [0, 1], 1: [2, 3]})
        ranks = [ParallelRankInfo.from_global_rank(rank, config) for rank in range(4)]
        # ranks (dp0,tp0) and (dp1,tp0) hold the same experts in different groups
        self.assertEqual(placement.experts_for_rank(ranks[0]), placement.experts_for_rank(ranks[2]))
        self.assertEqual([placement.ep_group_for_rank_info(r) for r in ranks], [0, 0, 1, 1])
        self.assertEqual([placement.ep_rank_for_rank_info(r) for r in ranks], [0, 1, 0, 1])
        self.assertEqual(placement.to_dict()["ep_group_count"], 2)

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
        config = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2, enable_expert_parallel=True)
        model = _MOE_MODEL
        placement = build_expert_placement(model.moe, config, total_layers=model.num_layers)
        rank0 = ParallelRankInfo.from_global_rank(0, config)
        rank1 = ParallelRankInfo.from_global_rank(1, config)

        cache0 = CacheConfig(16, _DEVICE, model, config, rank0, placement)
        cache1 = CacheConfig(16, _DEVICE, model, config, rank1, placement)

        self.assertEqual(cache0.size_per_token, cache1.size_per_token)
        self.assertEqual(cache0.local_kv_heads, cache1.local_kv_heads)
        self.assertLess(cache0.model_param_size, cache0.model_param_size_unsharded)
        # Each EP rank owns one of four experts, so routed expert bytes are a quarter.
        self.assertEqual(len(placement.experts_for_rank(rank0)), 1)

    def test_latency_records_moe_compute_and_imbalance(self):
        config = ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
        )
        model = _MOE_MODEL
        placement = build_expert_placement(model.moe, config, total_layers=model.num_layers)
        rank = ParallelRankInfo.from_global_rank(0, config)
        backend = OperatorTableLatencyBackend(
            device=_DEVICE,
            model=model,
            parallel_config=config,
            rank_info=rank,
            expert_placement=placement,
            fallback="analytical_only",
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
        self.assertEqual(backend.moe_layers_local, 2)
        self.assertEqual(backend.dense_layers_local, 2)
        self.assertTrue(backend.ep_enabled)
        # wide EP over 2 replicas: the group sees both replicas' tokens
        self.assertEqual(backend.moe_token_multiplier, 2)
        self.assertGreater(stats["moe_compute_latency"], 0)
        self.assertGreater(stats["moe_straggler_latency"], 0)
        self.assertGreater(stats["moe_expert_load_imbalance_ratio"], 1)
        self.assertGreater(backend.stats.component_seconds["moe"], 0)

    def test_per_dp_expert_parallelism_does_not_multiply_moe_tokens(self):
        model = _MOE_MODEL
        wide = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2, enable_expert_parallel=True)
        per_dp = wide.override(expert_parallel_scope="per_dp", expert_parallel_size=2)
        backends = {}
        for name, config in (("wide", wide), ("per_dp", per_dp)):
            backends[name] = OperatorTableLatencyBackend(
                device=_DEVICE,
                model=model,
                parallel_config=config,
                rank_info=ParallelRankInfo.from_global_rank(0, config),
                expert_placement=build_expert_placement(model.moe, config, total_layers=model.num_layers),
                fallback="analytical_only",
            )
        req = Request(0, prefill_len=16, decode_len=1, block_size=16)

        self.assertEqual(backends["wide"].moe_token_multiplier, 2)
        self.assertEqual(backends["per_dp"].moe_token_multiplier, 1)
        self.assertEqual((backends["wide"].moe_ep, backends["per_dp"].moe_ep), (4, 2))
        for backend in backends.values():
            self.assertGreater(backend.estimate_step_latency([req]), 0)
        # Both ranks route the same number of token-expert pairs (16 tokens x
        # top-2 over 2 local experts vs 32 tokens x top-2 over 1), but the
        # per-DP rank streams twice the expert weights.
        self.assertGreaterEqual(
            backends["per_dp"].stats.component_seconds["moe"],
            backends["wide"].stats.component_seconds["moe"],
        )
        self.assertEqual(backends["per_dp"].moe_stats.as_dict()["moe_ep_rank_count"], 2)

    def test_engine_runs_and_exports_moe_metrics(self):
        reset_g_time()
        env = simpy.Environment()
        hardware = test_hardware(_DEVICE, models=[_MOE_MODEL])
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
            raise AssertionError("moe request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        stats = get_moe_stats(engine)

        self.assertTrue(stats["effective_moe_config"]["is_moe_model"])
        self.assertGreater(stats["moe_compute_latency"], 0)
        self.assertGreater(stats["moe_all2all_event_count"], 0)
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
