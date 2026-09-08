from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

import simpy

from tests.test_af_executor import (
    AFRooflineStub,
    FixedComputeProvider,
    FixedTransferProvider,
    _psla_config,
)
from tests.test_af_parallel_planning import _cluster
from TokenSim.af.runtime import AFCacheConfig
from TokenSim.af.models import AFBlock, AFBlockPlacement
from TokenSim.af.planning import ParallelExecutionPlanner, RankLocalWork
from TokenSim.af.runtime import compile_af_runtime_plan, kv_head_partitions
from TokenSim.block.block import Device
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.cluster_config import ExecutionDomainConfig, LogicalInstanceConfig
from TokenSim.errors import ConfigurationError, OutOfBlocksError, SimulationStateError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.kv_transfer import P2PConnector
from TokenSim.moe.config import MoEModelConfig
from util.request import LLMSource
from util.results import get_af_stats


def _runtime_roofline(*, layers=2, capacity=0.1):
    roofline = AFRooflineStub(layers=layers)
    for hardware in roofline.hardwares.values():
        hardware.Capacity = capacity
    return roofline


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
    roofline=None,
    moe_config=None,
    dtype="fp16",
):
    cluster = cluster or _cluster()
    roofline = roofline or _runtime_roofline()
    return compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        roofline=roofline,
        model="TestModel",
        moe_config=moe_config or _moe_config(),
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
        roofline=_runtime_roofline(),
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
        roofline = AFRooflineStub()
        for hardware in roofline.hardwares.values():
            hardware.Capacity = 0.01
        engine = LLMEngine(
            env=simpy.Environment(),
            block_size=4,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=_psla_config(),
            cluster_config=cluster,
            roofline=roofline,
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
            roofline=_runtime_roofline(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
        )
        self.assertIsNone(engine.af_runtime_plan)


class AFRuntimePlanTest(unittest.TestCase):
    def test_combined_topology_compiles_every_rank_group_and_leader(self):
        plan = _runtime_plan()

        self.assertEqual(len(plan.rank_bindings), 12)
        a0 = plan.replica_for_instance("a-domain", "a0")
        a1 = plan.replica_for_instance("a-domain", "a1")
        f0 = plan.replica_for_instance("f-domain", "f0")
        self.assertEqual(a0.rank_keys, (("a-domain", 0), ("a-domain", 1)))
        self.assertEqual(a1.rank_keys, (("a-domain", 2), ("a-domain", 3)))
        self.assertEqual(a0.leader_rank_key, ("a-domain", 0))
        self.assertEqual(a1.leader_rank_key, ("a-domain", 2))
        self.assertEqual(len(f0.rank_keys), 4)
        self.assertEqual(plan.binding("f-domain", 0).tp_group, (0, 1))
        self.assertEqual(plan.binding("f-domain", 0).ep_group, (0, 2))
        self.assertEqual(plan.binding("f-domain", 2).expert_ids, (2, 3))
        self.assertTrue(all(state.model_bytes > 0 for state in plan.physical_states))

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
        self.assertEqual(a_leader.component_bytes()["embedding_lm_head"], 800000)
        self.assertEqual(a_leader.component_bytes()["router"], 128)
        self.assertNotIn("router", a_follower.component_bytes())
        self.assertNotIn("routed_expert", a_leader.component_bytes())
        self.assertNotIn("router", f_ep0.component_bytes())
        self.assertEqual(f_ep0.component_bytes()["dense_ffn"], 2048)
        self.assertEqual(f_ep0.component_bytes()["routed_expert"], 2048)
        self.assertEqual(f_ep0.component_bytes()["shared_expert"], 1024)
        self.assertNotIn("router", f_ep1.component_bytes())
        self.assertEqual(f_ep1.component_bytes()["dense_ffn"], 2048)
        self.assertEqual(f_ep1.component_bytes()["routed_expert"], 2048)
        self.assertEqual(f_ep1.component_bytes()["shared_expert"], 1024)
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

    def test_oom_and_missing_metadata_fail_before_execution(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            "domain/rank=.*physical_worker=.*model_bytes=.*capacity_bytes",
        ):
            _runtime_plan(roofline=_runtime_roofline(capacity=0.000001))

        roofline = _runtime_roofline()
        del roofline.models["TestModel"].Dmodel
        with self.assertRaisesRegex(ConfigurationError, "Dmodel is missing"):
            _runtime_plan(roofline=roofline)
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
        )
        dag = planner.build_layer(block, 0, cluster.af.links[0])
        expert_nodes = [node for node in dag.nodes if node.stage == "expert"]
        self.assertEqual({node.ep_rank for node in expert_nodes}, {1})
        for node in expert_nodes:
            binding = plan.binding(node.source_domain, node.domain_ranks[0])
            self.assertEqual(node.physical_workers, (binding.physical_identity,))
            self.assertIn(3, binding.expert_ids)

        bad = replace(
            expert_nodes[0],
            local_work=replace(expert_nodes[0].local_work, expert_ids=(0,)),
        )
        with self.assertRaisesRegex(SimulationStateError, "resident"):
            plan.validate_execution_node(bad)

    def test_runtime_summary_is_versioned_and_serializable(self):
        plan = _runtime_plan()
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        req = _request(9)
        manager.allocate(req)
        summary = plan.summary({replica.replica_key: manager})

        self.assertEqual(summary["schema"], "tokensim.af_runtime_state")
        self.assertEqual(summary["version"], "1.0")
        self.assertGreater(summary["replicas"][0]["logical_gpu_usage"]["used"], 0)
        self.assertEqual(summary["replicas"][0]["logical_cpu_usage"]["used"], 0)
        manager.swap_out(req)
        swapped = plan.summary({replica.replica_key: manager})
        self.assertGreater(swapped["replicas"][0]["logical_cpu_usage"]["used"], 0)
        json.dumps(summary)

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
        cluster = _cluster(a_tp=2, a_dp=2, f_tp=2, f_dp=2, f_ep=2)
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
        self.assertEqual(len(f_workers), 8)
        self.assertTrue(any(3 in worker.owned_expert_ids for worker in f_workers))
        operations = [event.operation for event in engine.af_executor.trace.events]
        self.assertEqual(operations.count("af_a_to_f"), 4)
        self.assertEqual(operations.count("af_f_to_a"), 4)
        runtime = get_af_stats(engine)["af_runtime_state"]
        self.assertEqual(runtime["schema"], "tokensim.af_runtime_state")


if __name__ == "__main__":
    unittest.main()
