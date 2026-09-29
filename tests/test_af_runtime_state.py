from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

import simpy

from tests.test_af_executor import (
    FixedComputeProvider,
    FixedTransferProvider,
    _psla_config,
    af_test_hardware,
)
from tests.test_af_parallel_planning import _cluster
from TokenSim.af.runtime import AFCacheConfig, AFTensorInterval
from TokenSim.af.models import AFBlock, AFBlockPlacement
from TokenSim.af.planning import ParallelExecutionPlanner, RankLocalWork
from TokenSim.af.runtime import compile_af_runtime_plan, kv_head_partitions
from TokenSim.block.block import Device
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.cluster_config import (
    DomainCollectiveConfig,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
)
from TokenSim.errors import ConfigurationError, OutOfBlocksError, SimulationStateError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.kv_transfer import P2PConnector
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment
from util.request import LLMSource
from util.results import get_af_stats


def _runtime_hardware(*, layers=2, capacity=0.1):
    """Catalog with ``capacity`` GiB per device and a ``layers``-layer TestModel."""
    return af_test_hardware(layers=layers, capacity_gib=capacity)


def _moe_config():
    return MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=1,
        num_moe_layers=1,
        moe_intermediate_size=32,
        num_shared_experts=1,
        intermediate_size=64,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=8,
    )


def _runtime_plan(
    cluster=None,
    *,
    hardware=None,
    moe_config=None,
    dtype="fp16",
):
    cluster = cluster or _cluster()
    hardware = hardware or _runtime_hardware()
    model = hardware.model("TestModel").with_moe_override(moe_config or _moe_config())
    return compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        model=model,
        hardware=hardware,
        block_size=4,
        dtype=dtype,
    )


def _request(request_id, prefill_len=8, decode_len=8):
    request = Request(request_id, prefill_len, decode_len, block_size=4)
    request.arrive(SimpleNamespace(now=float(request_id)))
    return request


def _engine(cluster, *, moe_config=None):
    env = simpy.Environment()
    psla = _psla_config()
    if moe_config is not None:
        psla._moe_config = moe_config
    engine = LLMEngine(
        env=env,
        block_size=4,
        batching="paged-attn",
        kv_transfer_config=KVTransferConfig.default(),
        psla_config=psla,
        cluster_config=cluster,
        hardware=_runtime_hardware(),
        prefill_worker_pool_type="round_robin",
        decode_worker_pool_type="round_robin",
        max_parallem_sum=8,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        af_compute_provider=FixedComputeProvider(0.001, 0.001),
        af_transfer_provider=FixedTransferProvider(
            0.001,
            0.001,
            dispatch=0.001,
            combine=0.001,
            collective=0.001,
        ),
    )
    return env, engine


def _split_phase_alias_cluster():
    base = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
    attention = base.execution_domains["a-domain"]
    ffn = base.execution_domains["f-domain"]
    prefill_attention = replace(
        attention,
        name="prefill-a",
        phases=("prefill",),
    )
    decode_attention = replace(
        attention,
        name="decode-a",
        phases=("decode",),
        logical_instances=[
            replace(instance, name=f"{instance.name}-decode")
            for instance in attention.logical_instances
        ],
    )
    link = base.af.links[0]
    af = replace(
        base.af,
        links=[
            replace(
                link,
                name="prefill-af",
                phases=("prefill",),
                attention_domain="prefill-a",
            ),
            replace(
                link,
                name="decode-af",
                phases=("decode",),
                attention_domain="decode-a",
            ),
        ],
        execution_domains=[prefill_attention, decode_attention, ffn],
    )
    return base.__class__(
        num_workers=base.num_workers,
        networks=base.networks,
        worker_groups=base.worker_groups,
        kv_transfer=base.kv_transfer,
        parallel_config=base.parallel_config,
        af=af,
    )


class AFRuntimeStateRegressionTest(unittest.TestCase):
    def test_domain_tp_builds_rank_local_state_when_global_tp_is_one(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        engine = LLMEngine(
            env=simpy.Environment(),
            block_size=4,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=_psla_config(),
            cluster_config=cluster,
            hardware=_runtime_hardware(layers=1, capacity=0.01),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            parallel_config=ParallelConfig(tensor_parallel_size=1),
            af_compute_provider=FixedComputeProvider(0.0, 0.0),
            af_transfer_provider=FixedTransferProvider(0.0, 0.0),
        )

        self.assertIsNotNone(engine.af_runtime_plan)
        a_bindings = engine.af_runtime_plan.bindings_for_instance("a-domain", "a0")
        self.assertEqual([binding.tp_rank for binding in a_bindings], [0, 1])
        self.assertEqual(
            [binding.is_scheduler_leader for binding in a_bindings],
            [True, False],
        )
        for binding in a_bindings:
            state = engine.af_runtime_plan.physical_state(binding.physical_identity)
            self.assertGreater(state.model_bytes, 0)
            self.assertGreater(state.kv_bytes_per_block, 0)

        self.assertEqual(len(engine.request_workers), 1)
        manager = engine.request_workers[0].scheduler.block_manager
        self.assertEqual(len(manager.rank_keys), 2)

    def test_split_phase_domains_share_equivalent_physical_scheduler_state(self):
        cluster = _split_phase_alias_cluster()
        _, engine = _engine(cluster)

        self.assertEqual(len(engine.request_workers), 1)
        manager = engine.request_workers[0].scheduler.block_manager
        self.assertIs(
            engine.af_replica_block_managers[("prefill-a", 0)],
            manager,
        )
        self.assertIs(
            engine.af_replica_block_managers[("decode-a", 0)],
            manager,
        )
        request = _request(11)
        manager.allocate(request)
        self.assertEqual(
            request.af_a_owners,
            {"prefill-a": "a0", "decode-a": "a0-decode"},
        )
        summary = engine.af_runtime_summary()
        replica_usage = {
            item["domain"]: item["logical_gpu_usage"]
            for item in summary["replicas"]
            if item["domain"] in {"prefill-a", "decode-a"}
        }
        self.assertEqual(
            replica_usage["prefill-a"],
            replica_usage["decode-a"],
        )

    def test_af_disabled_does_not_build_runtime_plan(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster.af.enabled = False
        cluster.worker_groups[0].capabilities = ("prefill", "decode_full")
        cluster.worker_groups[0].role = "hybrid"
        cluster.worker_groups = [cluster.worker_groups[0]]
        cluster.num_workers = 1
        engine = LLMEngine(
            env=simpy.Environment(),
            block_size=4,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=_psla_config(),
            cluster_config=cluster,
            hardware=_runtime_hardware(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
        )
        self.assertIsNone(engine.af_runtime_plan)


class AFRuntimePlanTest(unittest.TestCase):
    def test_combined_topology_compiles_every_rank_group_and_leader(self):
        plan = _runtime_plan()

        self.assertEqual(len(plan.rank_bindings), 8)
        a0 = plan.replica_for_instance("a-domain", "a0")
        a1 = plan.replica_for_instance("a-domain", "a1")
        f0 = plan.replica_for_instance("f-domain", "f0")
        f1 = plan.replica_for_instance("f-domain", "f1")
        self.assertEqual(a0.rank_keys, (("a-domain", 0), ("a-domain", 1)))
        self.assertEqual(a1.rank_keys, (("a-domain", 2), ("a-domain", 3)))
        self.assertEqual(a0.leader_rank_key, ("a-domain", 0))
        self.assertEqual(a1.leader_rank_key, ("a-domain", 2))
        self.assertEqual(f0.rank_keys, (("f-domain", 0), ("f-domain", 1)))
        self.assertEqual(f1.rank_keys, (("f-domain", 2), ("f-domain", 3)))
        self.assertEqual(plan.binding("f-domain", 0).tp_group, (0, 1))
        self.assertEqual(plan.binding("f-domain", 0).ep_group, (0, 1, 2, 3))
        self.assertEqual(plan.binding("f-domain", 2).ep_rank, 2)
        self.assertEqual(plan.binding("f-domain", 2).expert_ids, (2,))
        self.assertTrue(all(state.model_bytes > 0 for state in plan.physical_states))

    def test_ep_modes_reuse_four_f_states_and_resolve_execution_groups(self):
        for ep_size, mode, expert_tp in ((1, "expert_tp", 2), (4, "rank_reuse", 1)):
            with self.subTest(ep_size=ep_size):
                plan = _runtime_plan(
                    _cluster(a_tp=2, a_dp=2, f_tp=2, f_dp=2, f_ep=ep_size)
                )
                f_states = [
                    state
                    for state in plan.physical_states
                    if any(domain == "f-domain" for domain, _ in state.bindings)
                ]
                self.assertEqual(len(f_states), 4)
                self.assertEqual(
                    len({state.physical_identity for state in f_states}), 4
                )

                f_domain = plan.domain_runtime("f-domain")
                self.assertEqual(f_domain.expert_parallel_mode, mode)
                self.assertEqual(f_domain.expert_tensor_parallel_size, expert_tp)
                dense = plan.component_execution("f-domain", 1, "dense_ffn", 1)
                shared = plan.component_execution(
                    "f-domain", 0, "shared_expert", 1
                )
                self.assertEqual(
                    tuple(binding.domain_rank for binding in dense), (2, 3)
                )
                self.assertEqual(
                    tuple(binding.domain_rank for binding in shared), (2, 3)
                )

                routed = plan.expert_execution("f-domain", 3, 1)
                if ep_size == 1:
                    self.assertEqual(
                        tuple(binding.domain_rank for binding in routed), (2, 3)
                    )
                    self.assertTrue(
                        all(binding.ep_group == (2, 3) for binding in routed)
                    )
                    self.assertTrue(
                        all(3 in binding.expert_ids for binding in routed)
                    )
                else:
                    self.assertEqual(
                        tuple(binding.domain_rank for binding in routed), (3,)
                    )
                    self.assertEqual(routed[0].ep_group, (0, 1, 2, 3))
                    self.assertEqual(routed[0].ep_rank, 3)

    def test_local_shapes_use_exact_tp_axes_and_reject_ceil_shards(self):
        expert_tp = _runtime_plan(
            _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=2, f_ep=1)
        )
        reused = _runtime_plan(
            _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=2, f_ep=4)
        )

        self.assertEqual(
            expert_tp.local_shape("f-domain", 2, 0, "routed_expert"),
            {
                "global_hidden_size": 16,
                "hidden_size": 16,
                "tp_size": 2,
                "expert_tp_size": 2,
                "expert_tp_rank": 0,
                "intermediate_size": 16,
            },
        )
        self.assertEqual(
            reused.local_shape("f-domain", 3, 0, "routed_expert"),
            {
                "global_hidden_size": 16,
                "hidden_size": 16,
                "tp_size": 1,
                "expert_tp_size": 1,
                "expert_tp_rank": 0,
                "intermediate_size": 32,
            },
        )
        self.assertEqual(
            expert_tp.local_shape("f-domain", 2, 1, "dense_ffn")[
                "hidden_size"
            ],
            16,
        )
        self.assertEqual(
            expert_tp.local_shape("f-domain", 2, 0, "shared_expert"),
            {
                "global_hidden_size": 16,
                "hidden_size": 16,
                "tp_size": 2,
                "intermediate_size": 16,
            },
        )

        hardware = _runtime_hardware()
        model = replace(
            hardware.model("TestModel"),
            hidden_size=24,
        ).with_moe_override(replace(_moe_config(), hidden_size=24))
        with self.assertRaisesRegex(
            ConfigurationError, r"query heads=8.*A domain='a-domain' TP=3"
        ):
            compile_af_runtime_plan(
                cluster_config=_cluster(a_tp=3, a_dp=1, f_tp=1, f_dp=1, f_ep=1),
                worker_configs=_cluster(a_tp=3, a_dp=1, f_tp=1, f_dp=1, f_ep=1).workers(),
                hardware=hardware,
                model=model,
                block_size=4,
                dtype="fp16",
            )

        with self.assertRaisesRegex(ConfigurationError, "Routed Expert intermediate=33"):
            _runtime_plan(
                _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1),
                moe_config=replace(
                    _moe_config(),
                    moe_intermediate_size=33,
                    num_shared_experts=0,
                ),
            )
        with self.assertRaisesRegex(ConfigurationError, "Dense FFN intermediate=65"):
            _runtime_plan(
                _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1),
                moe_config=replace(_moe_config(), intermediate_size=65),
            )
        with self.assertRaisesRegex(ConfigurationError, "Shared FFN intermediate=33"):
            _runtime_plan(
                _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1),
                moe_config=replace(
                    _moe_config(),
                    moe_intermediate_size=33,
                    num_shared_experts=1,
                ),
            )
        rank_reuse_uneven_expert = _runtime_plan(
            _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=2),
            moe_config=replace(
                _moe_config(),
                moe_intermediate_size=33,
                num_shared_experts=0,
            ),
        )
        self.assertEqual(
            rank_reuse_uneven_expert.local_shape(
                "f-domain", 0, 0, "routed_expert"
            )["intermediate_size"],
            33,
        )

    def test_runtime_kv_replication_is_preserved_in_local_shapes(self):
        plan = _runtime_plan(
            _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1),
            moe_config=replace(_moe_config(), num_key_value_heads=1),
        )
        bindings = plan.bindings_for_instance("a-domain", "a0")

        self.assertEqual(
            [binding.kv_partition.global_head_ids for binding in bindings],
            [(0,), (0,)],
        )
        self.assertEqual(
            [binding.kv_partition.replica_factor for binding in bindings],
            [2, 2],
        )
        self.assertEqual(
            [
                plan.local_shape("a-domain", binding.domain_rank, 0, "attention")[
                    "num_kv_heads"
                ]
                for binding in bindings
            ],
            [1, 1],
        )

    def test_physical_ledger_deduplicates_alias_domain_components(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        original = cluster.execution_domains["a-domain"]
        cluster.af.execution_domains.append(
            ExecutionDomainConfig(
                name="a-domain-alias",
                capability="af_attention",
                phases=(("decode",)),
                worker_group=original.worker_group,
                parallel=original.parallel,
                logical_instances=[
                    LogicalInstanceConfig(
                        name="a0-alias",
                        rank_mapping=list(original.logical_instances[0].rank_mapping),
                    )
                ],
            )
        )
        baseline = _runtime_plan(_cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1))
        plan = _runtime_plan(cluster)
        baseline_state = baseline.physical_state("a-pool:0")
        state = plan.physical_state("a-pool:0")

        self.assertEqual(len(state.bindings), 2)
        self.assertEqual(state.model_bytes, baseline_state.model_bytes)
        self.assertEqual(
            len(state.components),
            len({component.component_id for component in state.components}),
        )

    def test_kv_heads_shard_replicate_and_reject_ambiguous_topology(self):
        sharded = kv_head_partitions(
            num_kv_heads=8,
            tensor_parallel_size=2,
            head_dim=2,
            num_layers=2,
            block_size=4,
            dtype_size_bytes=2,
        )
        replicated = kv_head_partitions(
            num_kv_heads=1,
            tensor_parallel_size=2,
            head_dim=2,
            num_layers=2,
            block_size=4,
            dtype_size_bytes=2,
        )

        self.assertEqual(
            [item.global_head_ids for item in sharded], [(0, 1, 2, 3), (4, 5, 6, 7)]
        )
        self.assertEqual([item.replica_factor for item in sharded], [1, 1])
        self.assertEqual([item.global_head_ids for item in replicated], [(0,), (0,)])
        self.assertEqual([item.replica_factor for item in replicated], [2, 2])
        with self.assertRaisesRegex(ConfigurationError, "cannot be evenly sharded"):
            kv_head_partitions(
                num_kv_heads=3,
                tensor_parallel_size=2,
                head_dim=2,
                num_layers=2,
                block_size=4,
                dtype_size_bytes=2,
            )

    def test_component_ownership_matches_hand_calculation(self):
        plan = _runtime_plan()
        a_leader = plan.physical_state("a-pool:0")
        a_follower = plan.physical_state("a-pool:1")
        f_ep0 = plan.physical_state("f-pool:0")
        f_ep1 = plan.physical_state("f-pool:2")

        self.assertEqual(a_leader.component_bytes()["attention"], 2048)
        self.assertEqual(a_leader.component_bytes()["embedding_lm_head"], 1600000)
        self.assertEqual(a_leader.component_bytes()["router"], 128)
        self.assertNotIn("router", a_follower.component_bytes())
        self.assertNotIn("routed_expert", a_leader.component_bytes())
        self.assertNotIn("router", f_ep0.component_bytes())
        self.assertEqual(f_ep0.component_bytes()["dense_ffn"], 3072)
        self.assertEqual(f_ep0.component_bytes()["routed_expert"], 3072)
        self.assertEqual(f_ep0.component_bytes()["shared_expert"], 1536)
        self.assertNotIn("router", f_ep1.component_bytes())
        self.assertEqual(f_ep1.component_bytes()["dense_ffn"], 3072)
        self.assertEqual(f_ep1.component_bytes()["routed_expert"], 3072)
        self.assertEqual(f_ep1.component_bytes()["shared_expert"], 1536)
        self.assertEqual(f_ep0.component_bytes(), f_ep1.component_bytes())

        a_bindings = [
            binding
            for binding in plan.rank_bindings
            if binding.capability == "af_attention"
        ]
        for binding in a_bindings:
            state = plan.physical_state(binding.physical_identity)
            if binding.is_scheduler_leader:
                self.assertIn("router", state.component_bytes())
            else:
                self.assertNotIn("router", state.component_bytes())
        for binding in plan.rank_bindings:
            if binding.capability == "af_ffn":
                self.assertNotIn(
                    "router",
                    plan.physical_state(binding.physical_identity).component_bytes(),
                )

        self.assertEqual([layer.layer_type for layer in plan.layers], ["moe", "dense"])
        for state in plan.physical_states:
            for component in state.components:
                if component.layer_id == 0:
                    self.assertNotEqual(component.category, "dense_ffn")
                if component.layer_id == 1:
                    self.assertNotIn(
                        component.category,
                        {"router", "routed_expert", "shared_expert"},
                    )
            self.assertEqual(
                len(state.components),
                len({component.component_id for component in state.components}),
            )

    def test_attention_memory_uses_actual_mha_gqa_and_mqa_widths(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        expected_bytes = {8: 1024, 2: 640, 1: 640}
        for kv_heads, expected in expected_bytes.items():
            with self.subTest(kv_heads=kv_heads):
                plan = _runtime_plan(
                    cluster,
                    hardware=_runtime_hardware(layers=1),
                    moe_config=replace(
                        _moe_config(),
                        num_moe_layers=1,
                        num_key_value_heads=kv_heads,
                    ),
                )
                state = plan.physical_state("a-pool:0")
                attention = next(
                    component
                    for component in state.components
                    if component.category == "attention"
                )
                self.assertEqual(attention.bytes, expected)
                self.assertEqual(dict(attention.shape)["kv_heads_local"], 4 if kv_heads == 8 else 1)
                self.assertIn("Q/K/V/O", attention.formula)

    def test_weight_and_kv_dtypes_change_independent_ledgers(self):
        baseline_cluster = _cluster(
            a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1
        )
        typed_cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        typed_cluster.af = replace(
            typed_cluster.af,
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp32",
        )
        typed_cluster.af.links = [
            replace(
                typed_cluster.af.links[0],
                a_to_f_dtype="fp8",
                f_to_a_dtype="fp16",
            )
        ]

        baseline = _runtime_plan(baseline_cluster, dtype="bf16")
        typed = _runtime_plan(typed_cluster, dtype="bf16")
        baseline_a = baseline.physical_state("a-pool:0")
        typed_a = typed.physical_state("a-pool:0")
        baseline_binding = baseline.binding("a-domain", 0)
        typed_binding = typed.binding("a-domain", 0)

        self.assertEqual(typed.effective_dtypes.weight_dtype, "fp8")
        self.assertEqual(typed.effective_dtypes.activation_dtype, "bf16")
        self.assertEqual(typed.effective_dtypes.kv_cache_dtype, "fp32")
        self.assertEqual(typed.effective_dtypes.a_to_f_dtype, "fp8")
        self.assertEqual(typed.effective_dtypes.f_to_a_dtype, "fp16")
        self.assertEqual(typed_a.model_bytes * 2, baseline_a.model_bytes)
        self.assertEqual(
            typed_binding.kv_partition.bytes_per_block,
            baseline_binding.kv_partition.bytes_per_block * 2,
        )

    def test_tensor_ownership_maps_mismatched_tp_by_global_intersections(self):
        cases = (
            (
                2,
                4,
                ((0, 0, 0, 4), (1, 0, 4, 8), (2, 1, 8, 12), (3, 1, 12, 16)),
            ),
            (
                4,
                2,
                ((0, 0, 0, 4), (0, 1, 4, 8), (1, 2, 8, 12), (1, 3, 12, 16)),
            ),
        )
        for a_tp, f_tp, expected in cases:
            with self.subTest(a_tp=a_tp, f_tp=f_tp):
                plan = _runtime_plan(
                    _cluster(
                        a_tp=a_tp,
                        a_dp=1,
                        f_tp=f_tp,
                        f_dp=1,
                        f_ep=f_tp,
                    )
                )
                a_owners = plan.ownerships_for_replica("a-domain", 0)
                f_owners = plan.ownerships_for_replica("f-domain", 0)
                self.assertEqual(
                    [(item.interval.start, item.interval.end) for item in a_owners],
                    [
                        (rank * (16 // a_tp), (rank + 1) * (16 // a_tp))
                        for rank in range(a_tp)
                    ],
                )
                self.assertEqual(
                    [(item.interval.start, item.interval.end) for item in f_owners],
                    [
                        (rank * (16 // f_tp), (rank + 1) * (16 // f_tp))
                        for rank in range(f_tp)
                    ],
                )
                self.assertTrue(all(item.retains_residual for item in a_owners))
                self.assertTrue(
                    all(not item.retains_residual for item in f_owners)
                )

                destinations = tuple(
                    item
                    for item in plan.partial_destinations
                    if item.phase == "decode"
                    and item.source_dp_rank == 0
                    and item.target_dp_rank == 0
                )
                actual = tuple(
                    (
                        item.source_rank,
                        item.target_rank,
                        item.interval.start,
                        item.interval.end,
                    )
                    for item in destinations
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    sum(item.interval.size for item in destinations),
                    plan.hidden_size,
                )
                self.assertEqual(
                    len(actual),
                    len(set(actual)),
                )

    def test_oom_and_missing_metadata_fail_before_execution(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            (
                "domain/rank=.*physical_worker=.*component_bytes=.*model_bytes=.*"
                "capacity_bytes=.*required_bytes=.*available_bytes"
            ),
        ):
            _runtime_plan(hardware=_runtime_hardware(capacity=0.000001))

        unknown_hardware = _cluster()
        for group in unknown_hardware.worker_groups:
            group.hardware = "missing-device"
        with self.assertRaisesRegex(
            ConfigurationError, "unknown hardware 'missing-device'"
        ):
            _runtime_plan(unknown_hardware)
        with self.assertRaisesRegex(ConfigurationError, "no explicit byte size"):
            _runtime_plan(dtype="mystery")

    def test_static_validation_rejects_missing_leader_and_bad_tp_group(self):
        plan = _runtime_plan()
        leader = plan.binding("a-domain", 0)
        no_leader = replace(
            plan,
            rank_bindings=tuple(
                (
                    replace(item, is_scheduler_leader=False)
                    if item.rank_key == leader.rank_key
                    else item
                )
                for item in plan.rank_bindings
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "one scheduler leader"):
            no_leader.validate()

        bad_group = replace(
            plan,
            rank_bindings=tuple(
                (
                    replace(item, tp_group=(item.domain_rank,))
                    if item.rank_key == leader.rank_key
                    else item
                )
                for item in plan.rank_bindings
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "TP group mismatch"):
            bad_group.validate()

        f_binding = plan.binding("f-domain", 0)
        bad_ep_group = replace(
            plan,
            rank_bindings=tuple(
                replace(item, ep_group=(item.domain_rank,))
                if item.rank_key == f_binding.rank_key
                else item
                for item in plan.rank_bindings
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "EP group mismatch"):
            bad_ep_group.validate()

        a_ownership = next(
            item for item in plan.tensor_ownerships if item.domain == "a-domain"
        )
        bad_residual = replace(
            plan,
            tensor_ownerships=tuple(
                replace(item, retains_residual=False)
                if item == a_ownership
                else item
                for item in plan.tensor_ownerships
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "retains_residual"):
            bad_residual.validate()

        destination = plan.partial_destinations[0]
        bad_destination = replace(
            plan,
            partial_destinations=(
                replace(destination, interval=AFTensorInterval(0, 1)),
                *plan.partial_destinations[1:],
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "expected"):
            bad_destination.validate()

    def test_tp_one_degenerates_to_one_local_shard(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        req = _request(20)
        manager.allocate(req)

        self.assertEqual(replica.logical_gpu_blocks, replica.local_gpu_blocks[0][1])
        self.assertTrue(
            all(
                len(mapping.shards) == 1
                for mapping in manager.mappings_for_request(req.id)
            )
        )


class ReplicaKVBlockManagerTest(unittest.TestCase):
    def _manager(self, gpu=(8, 8), cpu=(8, 8)):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        return ReplicaKVBlockManager(
            block_size=4,
            model="TestModel",
            replica=replica,
            gpu_blocks_by_rank=dict(zip(replica.rank_keys, gpu)),
            cpu_blocks_by_rank=dict(zip(replica.rank_keys, cpu)),
        )

    def test_allocate_append_and_free_map_one_shard_per_rank(self):
        manager = self._manager()
        req = _request(1)
        manager.allocate(req)

        mappings = manager.mappings_for_request(req.id)
        self.assertEqual(len(mappings), 2)
        self.assertTrue(all(len(item.shards) == 2 for item in mappings))
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )

        req.generation_idx = 1
        req._append_tokens(1)
        manager.append_slot(req)
        self.assertEqual(len(manager.mappings_for_request(req.id)), 3)
        manager.free(req)
        self.assertEqual(manager.mappings_for_request(req.id), ())
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_partial_rank_allocation_rolls_back_every_rank_and_request(self):
        manager = self._manager()
        req = _request(2)
        failing_rank = manager.rank_keys[1]

        def fail(_request):
            raise OutOfBlocksError("injected follower exhaustion")

        manager.rank_managers[failing_rank].allocate = fail
        with self.assertRaisesRegex(OutOfBlocksError, "injected"):
            manager.allocate(req)

        self.assertEqual(req.num_physical_token_blocks, 0)
        self.assertEqual(manager.mappings_for_request(req.id), ())
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_fork_reference_counts_and_last_free_are_group_consistent(self):
        manager = self._manager()
        parent = _request(3, prefill_len=4)
        child = _request(4, prefill_len=4)
        manager.allocate(parent)
        manager.fork(parent, child)

        for rank_key in manager.rank_keys:
            block = manager.rank_managers[rank_key].block_table.get_blocks(parent.id)[0]
            self.assertEqual(block.ref_count, 2)
        manager.release_request_blocks(parent)
        for rank_key in manager.rank_keys:
            block = manager.rank_managers[rank_key].block_table.get_blocks(child.id)[0]
            self.assertEqual(block.ref_count, 1)
        manager.release_request_blocks(child)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_swap_and_reservation_update_all_rank_pools(self):
        manager = self._manager()
        req = _request(5)
        manager.allocate(req)
        manager.swap_out(req)
        self.assertTrue(
            all(
                item.device == Device.CPU
                for item in manager.mappings_for_request(req.id)
            )
        )
        self.assertEqual(
            [manager.rank_status(key)["cpu_used"] for key in manager.rank_keys], [2, 2]
        )
        manager.swap_in(req)
        self.assertTrue(
            all(
                item.device == Device.GPU
                for item in manager.mappings_for_request(req.id)
            )
        )

        manager.reserve_gpu_blocks(1)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [3, 3]
        )
        manager.release_reserved_blocks(1)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )

    def test_partial_swap_failure_restores_every_rank_and_mapping(self):
        manager = self._manager()
        req = _request(13)
        manager.allocate(req)
        failing_rank = manager.rank_keys[1]

        def fail(_request):
            raise OutOfBlocksError("injected swap exhaustion")

        manager.rank_managers[failing_rank].swap_out = fail
        with self.assertRaisesRegex(OutOfBlocksError, "injected swap"):
            manager.swap_out(req)

        self.assertTrue(
            all(
                mapping.device == Device.GPU
                for mapping in manager.mappings_for_request(req.id)
            )
        )
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )
        self.assertEqual(
            [manager.rank_status(key)["cpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_logical_capacity_uses_smallest_rank(self):
        manager = self._manager(gpu=(100, 80), cpu=(40, 30))
        self.assertEqual(manager.num_total_gpu_blocks, 80)
        self.assertEqual(manager.num_total_cpu_blocks, 30)
        self.assertEqual(manager.get_gpu_status(), (80, 0, 80))

    def test_dynamic_invariants_report_missing_shard_and_owner_conflict(self):
        manager = self._manager()
        req = _request(6)
        manager.allocate(req)
        manager.logical_mappings.pop((req.id, 0))
        with self.assertRaisesRegex(SimulationStateError, "orphan KV shards"):
            manager.validate_invariants(req)

        manager._rebuild_request_mappings(req.id)
        req.af_a_owners[manager.domain] = "another-replica"
        with self.assertRaisesRegex(SimulationStateError, "owner conflict"):
            manager.validate_invariants(req)

    def test_scheduler_preemption_releases_all_rank_shards(self):
        manager = self._manager(gpu=(2, 2))
        cache = SimpleNamespace(
            block_size=4,
            num_gpu_blocks=2,
            num_cpu_blocks=8,
            model="TestModel",
        )
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=cache,
            max_parallem_sum=8,
            max_occupy_ratio=1,
            block_manager=manager,
        )
        req0 = _request(7, prefill_len=4, decode_len=4)
        req1 = _request(8, prefill_len=4, decode_len=4)
        scheduler.add_requests([req0, req1])
        scheduler.schedule_output()
        for req in (req0, req1):
            req.generation_idx = 4
            req._append_tokens(4)
        output = scheduler.schedule_output()
        self.assertTrue(output.preempted)
        victim = output.preempted[0]
        self.assertEqual(manager.mappings_for_request(victim.id), ())

    def test_prefix_reuse_keeps_logical_hits_and_rank_refcounts_aligned(self):
        manager = self._manager()
        original = Request(
            10,
            8,
            1,
            block_size=4,
            hash_ids=["a", "b"],
        )
        original.arrive(SimpleNamespace(now=0.0))
        manager.allocate(original)
        manager.commit_input_cache(original)
        manager.release_request_blocks(original)

        reused = Request(
            11,
            8,
            1,
            block_size=4,
            hash_ids=["a", "b"],
        )
        reused.arrive(SimpleNamespace(now=1.0))
        manager.allocate(reused)

        self.assertEqual(reused.reuse_hit_blocks, 2)
        self.assertEqual(len(manager.mappings_for_request(reused.id)), 2)
        for rank_key in manager.rank_keys:
            blocks = manager.rank_managers[rank_key].block_table.get_blocks(reused.id)
            self.assertTrue(all(block.ref_count == 1 for block in blocks))

    def test_connector_counts_logical_blocks_once_and_release_clears_all_shards(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        cache = AFCacheConfig.from_plan(plan, replica)
        connector = P2PConnector(KVTransferConfig(kv_connector="P2PConnector"), cache)
        req = _request(12)
        manager.allocate(req)

        transfer = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=0.1,
            request_block_counts=manager.get_request_block_counts([req]),
        )
        self.assertEqual(transfer.blocks, 2)
        self.assertEqual(transfer.request_block_counts, {req.id: 2})
        self.assertEqual(len(manager.mappings_for_request(req.id)), 2)
        manager.release_request_blocks(req)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys],
            [0, 0],
        )


class RuntimePlannerAndSummaryTest(unittest.TestCase):
    def test_planner_uses_runtime_endpoints_and_expert_residency(self):
        cluster = _cluster()
        plan = _runtime_plan(cluster)
        planner = ParallelExecutionPlanner(
            cluster,
            hidden_size=16,
            dtype_size_bytes=2,
            num_experts=4,
            top_k=1,
            runtime_plan=plan,
        )
        placement = planner.placement(
            "decode",
            "a-pool:0",
            cluster.execution_domains["f-domain"].logical_instances[0],
        )
        original_tokens = tuple((1, token_id) for token_id in range(4))
        snapshot = LayerRoutingSnapshot(
            iteration_id=1,
            phase="decode",
            decode_step=1,
            layer_id=0,
            top_k=1,
            original_tokens=original_tokens,
            assignments=tuple(
                RoutingAssignment(
                    iteration_id=1,
                    phase="decode",
                    decode_step=1,
                    layer_id=0,
                    request_id=request_id,
                    token_id=token_id,
                    top_k_index=0,
                    expert_id=3,
                    routing_weight=1.0,
                    a_dp_rank=placement.a_dp_rank,
                    a_tp_rank=0,
                )
                for request_id, token_id in original_tokens
            ),
        )
        block = AFBlock(
            iteration_id=1,
            block_id=1,
            phase="decode",
            request_ids=(1,),
            request_count=1,
            token_count=4,
            context_lengths=(4,),
            source_a_worker_id=0,
            link_name="af-link",
            placement=placement,
            expert_histogram=((3, 4),),
            routing_snapshots=(snapshot,),
        )
        dag = planner.build_layer(block, 0, cluster.af.links[0])
        expert_nodes = [
            node for node in dag.nodes if node.stage == "routed_expert"
        ]
        self.assertEqual({node.ep_rank for node in expert_nodes}, {3})
        for node in expert_nodes:
            binding = plan.binding(node.source_domain, node.domain_ranks[0])
            self.assertEqual(node.physical_workers, (binding.physical_identity,))
            self.assertIn(3, binding.expert_ids)

        a_to_f = [node for node in dag.nodes if node.operation == "af_a_to_f"]
        self.assertEqual({node.target_layout for node in a_to_f}, {"replicated"})
        self.assertEqual({node.source_layout for node in a_to_f}, {"replicated"})
        by_f_target = {}
        for node in a_to_f:
            by_f_target.setdefault(node.target_endpoint.domain_rank, 0)
            by_f_target[node.target_endpoint.domain_rank] += node.payload_bytes
            self.assertTrue(
                all(item.interval == (0, 16) for item in node.tensor_slices)
            )
        self.assertEqual(by_f_target, {0: 4 * 16 * 2, 1: 4 * 16 * 2, 3: 4 * 16 * 2})

        combine_nodes = [
            node for node in dag.nodes if node.stage == "weighted_combine"
        ]
        self.assertEqual(len(combine_nodes), 2)
        for node in combine_nodes:
            self.assertEqual(len(node.local_work.routing_edges), 4)
            self.assertEqual(
                {
                    (
                        edge.request_id,
                        edge.token_id,
                        edge.expert_id,
                        edge.top_k_index,
                        edge.routing_weight,
                    )
                    for edge in node.local_work.routing_edges
                },
                {(1, token_id, 3, 0, 1.0) for token_id in range(4)},
            )
            self.assertEqual(
                dict(node.local_work.tensor_shape)["hidden_size"], 8
            )

        f_to_a = [node for node in dag.nodes if node.operation == "af_f_to_a"]
        self.assertEqual({node.target_layout for node in f_to_a}, {"partial"})
        self.assertTrue(
            all(
                all(item.interval in {(0, 8), (8, 16)} for item in node.tensor_slices)
                for node in f_to_a
            )
        )

        bad = replace(
            expert_nodes[0],
            local_work=replace(expert_nodes[0].local_work, expert_ids=(0,)),
        )
        with self.assertRaisesRegex(SimulationStateError, "resident"):
            plan.validate_execution_node(bad)

    def test_collective_operation_controls_layout_and_logical_bytes(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        cluster.execution_domains["a-domain"].communication_plan = [
            DomainCollectiveConfig(
                operation="all_reduce",
                after_stage="a_compute",
                participants=(0, 1),
                network="af-net",
                resource_group="af-fabric",
                resource_lane="a_to_f",
            ),
            DomainCollectiveConfig(
                operation="all_gather",
                after_stage="residual",
                participants=(0, 1),
                network="af-net",
                resource_group="af-fabric",
                resource_lane="f_to_a",
            ),
        ]
        cluster.execution_domains["f-domain"].communication_plan = [
            DomainCollectiveConfig(
                operation="reduce_scatter",
                after_stage="expert_compute",
                participants=(0, 1),
                network="af-net",
                resource_group="af-fabric",
                resource_lane="f_to_a",
            )
        ]
        plan = _runtime_plan(cluster)
        planner = ParallelExecutionPlanner(
            cluster,
            hidden_size=16,
            dtype_size_bytes=2,
            num_experts=4,
            top_k=1,
            runtime_plan=plan,
        )
        placement = planner.placement(
            "decode",
            "a-pool:0",
            cluster.execution_domains["f-domain"].logical_instances[0],
        )
        block = AFBlock(
            iteration_id=7,
            block_id=0,
            phase="decode",
            request_ids=(1,),
            request_count=1,
            token_count=1,
            context_lengths=(4,),
            source_a_worker_id=0,
            link_name="af-link",
            placement=placement,
        )
        dag = planner.build_layer(block, 1, cluster.af.links[0])
        by_operation = {
            node.operation: node
            for node in dag.nodes
            if node.operation in {"all_reduce", "all_gather", "reduce_scatter"}
        }
        self.assertEqual(set(by_operation), {"all_reduce", "all_gather", "reduce_scatter"})
        expected = {
            "all_reduce": (32, 32, "partial", "replicated"),
            "all_gather": (16, 32, "sharded", "replicated"),
            "reduce_scatter": (32, 16, "partial", "sharded"),
        }
        for operation, (input_bytes, output_bytes, source_layout, target_layout) in expected.items():
            node = by_operation[operation]
            self.assertEqual(node.source_layout, source_layout)
            self.assertEqual(node.target_layout, target_layout)
            self.assertEqual(
                dict(node.logical_input_bytes_per_rank),
                {0: input_bytes, 1: input_bytes},
            )
            self.assertEqual(
                dict(node.logical_output_bytes_per_rank),
                {0: output_bytes, 1: output_bytes},
            )

        a_to_f = [node for node in dag.nodes if node.operation == "af_a_to_f"]
        self.assertEqual({node.source_layout for node in a_to_f}, {"replicated"})
        self.assertEqual({node.target_layout for node in a_to_f}, {"replicated"})
        bytes_per_f_rank = {}
        for node in a_to_f:
            bytes_per_f_rank.setdefault(node.target_endpoint.domain_rank, 0)
            bytes_per_f_rank[node.target_endpoint.domain_rank] += node.payload_bytes
        self.assertEqual(bytes_per_f_rank, {0: 32, 1: 32})

        f_to_a = [node for node in dag.nodes if node.operation == "af_f_to_a"]
        self.assertEqual({node.source_layout for node in f_to_a}, {"sharded"})
        self.assertEqual({node.target_layout for node in f_to_a}, {"sharded"})
        bytes_per_a_rank = {}
        for node in f_to_a:
            bytes_per_a_rank.setdefault(node.target_endpoint.domain_rank, 0)
            bytes_per_a_rank[node.target_endpoint.domain_rank] += node.payload_bytes
        self.assertEqual(bytes_per_a_rank, {0: 16, 1: 16})

    def test_collective_layout_mismatch_fails_before_execution(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        a_domain = cluster.execution_domains["a-domain"]
        a_domain.communication_plan[0] = DomainCollectiveConfig(
            operation="reduce_scatter",
            after_stage="a_compute",
            participants=(0, 1),
            network="af-net",
            resource_group="af-fabric",
            resource_lane="a_to_f",
        )
        plan = _runtime_plan(cluster)
        planner = ParallelExecutionPlanner(
            cluster,
            hidden_size=16,
            dtype_size_bytes=2,
            num_experts=4,
            top_k=1,
            runtime_plan=plan,
        )
        placement = planner.placement(
            "decode",
            "a-pool:0",
            cluster.execution_domains["f-domain"].logical_instances[0],
        )
        snapshot = LayerRoutingSnapshot(
            iteration_id=8,
            phase="decode",
            decode_step=1,
            layer_id=0,
            top_k=1,
            original_tokens=((1, 3),),
            assignments=(
                RoutingAssignment(
                    iteration_id=8,
                    phase="decode",
                    decode_step=1,
                    layer_id=0,
                    request_id=1,
                    token_id=3,
                    top_k_index=0,
                    expert_id=0,
                    routing_weight=1.0,
                    a_dp_rank=0,
                    a_tp_rank=0,
                ),
            ),
        )
        moe_block = AFBlock(
            iteration_id=8,
            block_id=0,
            phase="decode",
            request_ids=(1,),
            request_count=1,
            token_count=1,
            context_lengths=(4,),
            source_a_worker_id=0,
            link_name="af-link",
            placement=placement,
            routing_snapshots=(snapshot,),
        )
        with self.assertRaisesRegex(ConfigurationError, "downstream requires"):
            planner.build_layer(moe_block, 0, cluster.af.links[0])

        dense_cluster = _cluster(
            a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1
        )
        dense_cluster.execution_domains["f-domain"].communication_plan[0] = (
            DomainCollectiveConfig(
                operation="all_gather",
                after_stage="expert_compute",
                participants=(0, 1),
                network="af-net",
                resource_group="af-fabric",
                resource_lane="f_to_a",
            )
        )
        dense_plan = _runtime_plan(dense_cluster)
        dense_planner = ParallelExecutionPlanner(
            dense_cluster,
            hidden_size=16,
            dtype_size_bytes=2,
            num_experts=4,
            top_k=1,
            runtime_plan=dense_plan,
        )
        dense_placement = dense_planner.placement(
            "decode",
            "a-pool:0",
            dense_cluster.execution_domains["f-domain"].logical_instances[0],
        )
        dense_block = replace(
            moe_block,
            placement=dense_placement,
            routing_snapshots=(),
        )
        with self.assertRaisesRegex(ConfigurationError, "producer emits 'partial'"):
            dense_planner.build_layer(dense_block, 1, dense_cluster.af.links[0])

    def test_runtime_summary_is_versioned_and_serializable(self):
        plan = _runtime_plan()
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        req = _request(9)
        manager.allocate(req)
        summary = plan.summary({replica.replica_key: manager})

        self.assertEqual(summary["schema"], "tokensim.af_runtime_state")
        self.assertEqual(summary["version"], "2.0")
        self.assertGreater(summary["replicas"][0]["logical_gpu_usage"]["used"], 0)
        self.assertEqual(summary["replicas"][0]["logical_cpu_usage"]["used"], 0)
        manager.swap_out(req)
        swapped = plan.summary({replica.replica_key: manager})
        self.assertGreater(swapped["replicas"][0]["logical_cpu_usage"]["used"], 0)
        rebuilt = json.loads(json.dumps(summary))
        self.assertEqual(len(rebuilt["rank_bindings"]), len(plan.rank_bindings))
        self.assertEqual(
            len(rebuilt["physical_workers"]), len(plan.physical_states)
        )
        f_domain = next(
            item for item in rebuilt["domains"] if item["name"] == "f-domain"
        )
        self.assertEqual(f_domain["base_world_size"], 4)
        self.assertEqual(f_domain["expert_parallel_mode"], "rank_reuse")
        self.assertEqual(f_domain["expert_tensor_parallel_size"], 1)
        f_bindings = [
            item for item in rebuilt["rank_bindings"] if item["domain"] == "f-domain"
        ]
        self.assertEqual(
            {tuple(item["ep_group"]) for item in f_bindings},
            {(0, 1, 2, 3)},
        )
        for state in rebuilt["physical_workers"]:
            self.assertEqual(
                state["model_bytes"],
                sum(component["bytes"] for component in state["components"]),
            )
        self.assertEqual(
            rebuilt["effective_dtypes"]["bytes_per_element"]["weight_dtype"],
            2,
        )

    def test_dp_replicas_have_independent_cache_and_stable_request_owner(self):
        cluster = _cluster(a_tp=2, a_dp=2, f_tp=1, f_dp=1, f_ep=1)
        _, engine = _engine(cluster)
        managers = engine.af_replica_block_managers
        manager0 = managers[("a-domain", 0)]
        manager1 = managers[("a-domain", 1)]
        req0 = _request(30)
        req1 = _request(31)
        manager0.allocate(req0)
        manager1.allocate(req1)

        self.assertEqual(req0.af_a_owners, {"a-domain": "a0"})
        self.assertEqual(req1.af_a_owners, {"a-domain": "a1"})
        self.assertEqual(manager0.get_gpu_status()[1], 2)
        self.assertEqual(manager1.get_gpu_status()[1], 2)
        manager0.release_request_blocks(req0)
        self.assertEqual(manager0.get_gpu_status()[1], 0)
        self.assertEqual(manager1.get_gpu_status()[1], 2)

        req1.status = RequestStatus.RUNNING
        with self.assertRaisesRegex(SimulationStateError, "owner conflict"):
            manager0.allocate(req1)

    def test_combined_tp_dp_ep_engine_runs_with_one_scheduler_per_a_replica(self):
        cluster = _cluster(a_tp=2, a_dp=2, f_tp=2, f_dp=2, f_ep=4)
        env, engine = _engine(cluster, moe_config=_moe_config())
        requests = [
            Request(
                request_id,
                4,
                1,
                block_size=4,
                prefill_expert_histogram={0: 1, 3: 3},
            )
            for request_id in (40, 41)
        ]
        env.process(LLMSource(env, engine, requests, qps=1, distribution="burst"))

        # Use the public task enum without reaching through the engine instance.
        from TokenSim.llm.llm_engine import Task

        def stop_when_done():
            while env.now < 1:
                if all(request.is_done for request in requests):
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return
                yield env.timeout(1e-5)
            raise AssertionError("combined AF requests did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)

        self.assertEqual(len(engine.request_workers), 2)
        self.assertEqual({worker.dp_rank for worker in engine.request_workers}, {0, 1})
        self.assertTrue(all(request.is_done for request in requests))
        self.assertEqual(
            {tuple(request.af_a_owners.items()) for request in requests},
            {(("a-domain", "a0"),), (("a-domain", "a1"),)},
        )
        f_workers = [
            worker
            for worker in engine.workers
            if any(
                binding.capability == "af_ffn" for binding in worker.af_runtime_bindings
            )
        ]
        self.assertEqual(len(f_workers), 4)
        self.assertTrue(any(3 in worker.owned_expert_ids for worker in f_workers))
        operations = [event.operation for event in engine.af_executor.trace.events]
        self.assertEqual(operations.count("af_a_to_f"), 4)
        self.assertEqual(operations.count("af_f_to_a"), 4)
        runtime = get_af_stats(engine)["af_runtime_state"]
        self.assertEqual(runtime["schema"], "tokensim.af_runtime_state")


if __name__ == "__main__":
    unittest.main()
