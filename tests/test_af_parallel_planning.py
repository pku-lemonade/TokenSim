import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

from TokenSim.af.models import AFBlock, AFBlockPlacement
from TokenSim.af.planning import (
    AFExecutionDAG,
    ExecutionNode,
    GlobalRoundRobinReplicaSelector,
    ParallelExecutionPlanner,
    RankLocalWork,
)
from TokenSim.af.runtime import compile_af_runtime_plan
from TokenSim.config.cluster_config import ClusterConfig, DomainCollectiveConfig
from TokenSim.errors import ConfigurationError
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment


def _replica(group, name, dp_rank, tp_size, ep_size, worker_offset):
    return {
        "name": name,
        "rank_mapping": [
            {
                "domain_rank": dp_rank * tp_size + tp_rank,
                "tp_rank": tp_rank,
                "pp_rank": 0,
                "dp_rank": dp_rank,
                "ep_rank": dp_rank * tp_size + tp_rank if ep_size > 1 else 0,
                "physical_worker": {
                    "group_name": group,
                    "worker_index": worker_offset + tp_rank,
                },
            }
            for tp_rank in range(tp_size)
        ],
    }


def _cluster(
    a_tp=2,
    a_dp=2,
    f_tp=2,
    f_dp=2,
    f_ep=None,
    expert_placement_strategy="linear",
):
    f_ep = f_tp * f_dp if f_ep is None else f_ep
    a_workers = a_tp * a_dp
    f_workers = f_tp * f_dp
    return ClusterConfig(
        num_workers=a_workers + f_workers,
        networks={"af-net": "test"},
        worker_groups=[
            {
                "name": "a-pool",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": a_workers,
                "network": "af-net",
            },
            {
                "name": "f-pool",
                "capabilities": ["af_ffn"],
                "hardware": "f-device",
                "num_workers": f_workers,
                "network": "af-net",
            },
        ],
        af={
            "enabled": True,
            "phases": ["prefill", "decode"],
            "af_block_size": 1,
            "links": [
                {
                    "name": "af-link",
                    "phases": ["prefill", "decode"],
                    "attention_group": "a-pool",
                    "ffn_group": "f-pool",
                    "attention_domain": "a-domain",
                    "ffn_domain": "f-domain",
                    "network": "af-net",
                    "resource_group": "af-fabric",
                    "duplex": "full",
                    "a_to_f_capacity": 1,
                    "f_to_a_capacity": 1,
                }
            ],
            "execution_domains": [
                {
                    "name": "a-domain",
                    "capability": "af_attention",
                    "phases": ["prefill", "decode"],
                    "worker_group": "a-pool",
                    "parallel": {
                        "tensor_parallel_size": a_tp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": a_dp,
                    },
                    "logical_instances": [
                        _replica("a-pool", f"a{rank}", rank, a_tp, 1, rank * a_tp)
                        for rank in range(a_dp)
                    ],
                    "communication_plan": (
                        [
                            {
                                "operation": "all_reduce",
                                "after_stage": "a_compute",
                                "participants": list(range(a_tp)),
                                "network": "af-net",
                                "resource_group": "af-fabric",
                                "resource_lane": "a_to_f",
                            },
                            {
                                "operation": "all_gather",
                                "after_stage": "residual",
                                "participants": list(range(a_tp)),
                                "network": "af-net",
                                "resource_group": "af-fabric",
                                "resource_lane": "f_to_a",
                            },
                        ]
                        if a_tp > 1
                        else []
                    ),
                },
                {
                    "name": "f-domain",
                    "capability": "af_ffn",
                    "phases": ["prefill", "decode"],
                    "worker_group": "f-pool",
                    "parallel": {
                        "tensor_parallel_size": f_tp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": f_dp,
                        "expert_parallel_size": f_ep,
                        "expert_placement_strategy": expert_placement_strategy,
                    },
                    "logical_instances": [
                        _replica(
                            "f-pool",
                            f"f{rank}",
                            rank,
                            f_tp,
                            f_ep,
                            rank * f_tp,
                        )
                        for rank in range(f_dp)
                    ],
                    "communication_plan": (
                        [
                            {
                                "operation": "all_reduce",
                                "after_stage": "expert_compute",
                                "participants": list(range(f_tp)),
                                "network": "af-net",
                                "resource_group": "af-fabric",
                                "resource_lane": "f_to_a",
                            }
                        ]
                        if f_tp > 1
                        else []
                    ),
                },
            ],
        },
    )


def _block(histogram=((0, 10), (2, 2))):
    return AFBlock(
        iteration_id=3,
        block_id=4,
        phase="decode",
        request_ids=(1,),
        request_count=1,
        token_count=12,
        context_lengths=(32,),
        source_a_worker_id=2,
        link_name="af-link",
        placement=AFBlockPlacement(
            a_domain="a-domain",
            a_instance="a1",
            a_dp_rank=1,
            f_domain="f-domain",
            f_instance="f0",
            f_dp_rank=0,
        ),
        expert_histogram=histogram,
    )


def _formal_runtime_plan(cluster, moe_config):
    hardware = SimpleNamespace(Capacity=0.1, MM_Card_Num=1)
    roofline = SimpleNamespace(
        hardwares={"a-device": hardware, "f-device": hardware},
        models={
            "TestModel": SimpleNamespace(
                Name="TestModel",
                Nhead=8,
                Dmodel=16,
                Nlayer=2,
                Multi_Query=False,
                Grouped_Query=False,
                FFN_Hidden=64,
            )
        },
    )
    return compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        roofline=roofline,
        model="TestModel",
        moe_config=moe_config,
        block_size=4,
        dtype="fp16",
    )


def _small_formal_moe_dag(cluster=None):
    cluster = cluster or _cluster(a_tp=1, a_dp=1, f_tp=2, f_dp=1, f_ep=2)
    moe_config = MoEModelConfig(
        is_moe_model=True,
        num_experts=2,
        num_experts_per_tok=1,
        num_moe_layers=1,
        moe_intermediate_size=32,
        num_shared_experts=1,
        intermediate_size=64,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=8,
    )
    runtime_plan = _formal_runtime_plan(cluster, moe_config)
    planner = ParallelExecutionPlanner(
        cluster,
        hidden_size=16,
        dtype_size_bytes=2,
        num_experts=2,
        top_k=1,
        runtime_plan=runtime_plan,
    )
    placement = planner.placement(
        "decode",
        "a-pool:0",
        cluster.execution_domains["f-domain"].logical_instances[0],
    )
    assignment = RoutingAssignment(
        iteration_id=6,
        phase="decode",
        decode_step=1,
        layer_id=0,
        request_id=1,
        token_id=8,
        top_k_index=0,
        expert_id=0,
        routing_weight=1.0,
        a_dp_rank=0,
        a_tp_rank=0,
    )
    snapshot = LayerRoutingSnapshot(
        iteration_id=6,
        phase="decode",
        decode_step=1,
        layer_id=0,
        top_k=1,
        original_tokens=((1, 8),),
        assignments=(assignment,),
    )
    block = AFBlock(
        iteration_id=6,
        block_id=0,
        phase="decode",
        request_ids=(1,),
        request_count=1,
        token_count=1,
        context_lengths=(9,),
        source_a_worker_id=0,
        link_name="af-link",
        placement=placement,
        routing_snapshots=(snapshot,),
    )
    return planner.build_layer(block, 0, cluster.af.links[0])


class DAGValidationTest(unittest.TestCase):
    def _node(self, node_id, predecessors=(), workers=("a-pool:0",)):
        return ExecutionNode(
            node_id=node_id,
            operation="a_rank_compute",
            kind="compute",
            predecessor_ids=predecessors,
            stage="a",
            direction=None,
            source_domain="a-domain",
            target_domain="a-domain",
            logical_participants=("a0",),
            domain_ranks=(0,),
            physical_workers=workers,
            dp_rank=0,
            local_work=RankLocalWork(1),
        )

    def test_dag_rejects_duplicate_missing_cyclic_and_multiworker_compute(self):
        node = self._node("n0")
        with self.assertRaisesRegex(ConfigurationError, "duplicate node"):
            AFExecutionDAG((node, node))
        with self.assertRaisesRegex(ConfigurationError, "missing or cyclic"):
            AFExecutionDAG((self._node("n1", ("n0",)),))
        with self.assertRaisesRegex(ConfigurationError, "exactly one physical worker"):
            self._node("n2", workers=("a-pool:0", "a-pool:1"))

    def test_runtime_dag_rejects_semantic_corruption_before_execution(self):
        dag = _small_formal_moe_dag()

        def rebuild(original, replacement):
            return AFExecutionDAG(
                tuple(replacement if node is original else node for node in dag.nodes)
            )

        routed = next(node for node in dag.nodes if node.stage == "routed_expert")
        edge = routed.local_work.routing_edges[0]
        duplicate_work = replace(
            routed.local_work,
            assignment_count=2,
            routing_edges=(edge, edge),
        )
        with self.assertRaisesRegex(ConfigurationError, "consumes an assignment twice"):
            rebuild(routed, replace(routed, local_work=duplicate_work))

        missing_work = replace(
            routed.local_work,
            assignment_count=0,
            routing_edges=(),
        )
        with self.assertRaisesRegex(ConfigurationError, "missing=1"):
            rebuild(routed, replace(routed, local_work=missing_work))

        shared = next(node for node in dag.nodes if node.stage == "shared_expert")
        bad_shared_work = replace(
            shared.local_work,
            token_count=0,
            original_token_count=0,
            output_token_count=0,
        )
        with self.assertRaisesRegex(ConfigurationError, "every original token"):
            rebuild(shared, replace(shared, local_work=bad_shared_work))

        residual = next(node for node in dag.nodes if node.stage == "residual")
        with self.assertRaisesRegex(ConfigurationError, "must execute in A domain"):
            rebuild(
                residual,
                replace(
                    residual,
                    source_domain="f-domain",
                    target_domain="f-domain",
                ),
            )
        bad_residual_work = replace(residual.local_work, token_count=2)
        with self.assertRaisesRegex(ConfigurationError, "original token count"):
            rebuild(residual, replace(residual, local_work=bad_residual_work))
        duplicate_residual = replace(
            residual,
            node_id=f"{residual.node_id}:duplicate",
        )
        with self.assertRaisesRegex(ConfigurationError, "exactly once"):
            AFExecutionDAG(dag.nodes + (duplicate_residual,))

        combine = next(node for node in dag.nodes if node.stage == "weighted_combine")
        bad_combine_work = replace(combine.local_work, output_token_count=0)
        with self.assertRaisesRegex(ConfigurationError, "produce the original"):
            rebuild(combine, replace(combine, local_work=bad_combine_work))

        transfer = next(node for node in dag.nodes if node.operation == "af_a_to_f")
        with self.assertRaisesRegex(ConfigurationError, "duplicate physical tensor copy"):
            rebuild(
                transfer,
                replace(
                    transfer,
                    tensor_slices=(transfer.tensor_slices[0],) * 2,
                ),
            )

        router = next(node for node in dag.nodes if node.stage == "router")
        with self.assertRaisesRegex(ConfigurationError, "terminal nodes must equal"):
            AFExecutionDAG(dag.nodes, terminal_node_ids=(router.node_id,))


class ParallelExecutionPlannerTest(unittest.TestCase):
    def test_required_collective_covers_the_complete_tp_execution_branch(self):
        for after_stage in ("a_compute", "residual"):
            with self.subTest(after_stage=after_stage):
                cluster = _cluster(
                    a_tp=4,
                    a_dp=1,
                    f_tp=1,
                    f_dp=1,
                    f_ep=1,
                )
                a_domain = cluster.execution_domains["a-domain"]
                a_domain.communication_plan = [
                    replace(
                        plan,
                        participants=(0, 2),
                    )
                    if plan.after_stage == after_stage
                    else plan
                    for plan in a_domain.communication_plan
                ]
                with self.assertRaisesRegex(
                    ConfigurationError, "cover the complete execution branch"
                ):
                    _small_formal_moe_dag(cluster)

    def test_formal_dense_and_moe_dags_preserve_ownership_casts_and_workloads(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=4, f_dp=1, f_ep=4)
        cluster.af.links = [
            replace(
                cluster.af.links[0],
                a_to_f_dtype="fp8",
                f_to_a_dtype="fp8",
            )
        ]
        cluster.execution_domains["f-domain"].communication_plan = [
            DomainCollectiveConfig(
                operation="reduce_scatter",
                after_stage="expert_compute",
                participants=(0, 1, 2, 3),
                network="af-net",
                resource_group="af-fabric",
                resource_lane="f_to_a",
            )
        ]
        moe_config = MoEModelConfig(
            is_moe_model=True,
            num_experts=4,
            num_experts_per_tok=2,
            num_moe_layers=1,
            first_k_dense_replace=1,
            moe_intermediate_size=32,
            num_shared_experts=1,
            intermediate_size=64,
            hidden_size=16,
            num_attention_heads=8,
            num_key_value_heads=8,
        )
        runtime_plan = _formal_runtime_plan(cluster, moe_config)
        planner = ParallelExecutionPlanner(
            cluster,
            hidden_size=16,
            dtype_size_bytes=2,
            num_experts=4,
            top_k=2,
            runtime_plan=runtime_plan,
        )
        placement = planner.placement(
            "decode",
            "a-pool:0",
            cluster.execution_domains["f-domain"].logical_instances[0],
        )
        tokens = ((1, 8), (2, 10))
        assignments = tuple(
            RoutingAssignment(
                iteration_id=5,
                phase="decode",
                decode_step=1,
                layer_id=1,
                request_id=request_id,
                token_id=token_id,
                top_k_index=top_k_index,
                expert_id=top_k_index,
                routing_weight=(0.75 if top_k_index == 0 else 0.25),
                a_dp_rank=0,
                a_tp_rank=0,
            )
            for request_id, token_id in tokens
            for top_k_index in range(2)
        )
        snapshot = LayerRoutingSnapshot(
            iteration_id=5,
            phase="decode",
            decode_step=1,
            layer_id=1,
            top_k=2,
            original_tokens=tokens,
            assignments=assignments,
        )
        block = AFBlock(
            iteration_id=5,
            block_id=0,
            phase="decode",
            request_ids=(1, 2),
            request_count=2,
            token_count=2,
            context_lengths=(9, 11),
            source_a_worker_id=0,
            link_name="af-link",
            placement=placement,
            routing_snapshots=(snapshot,),
        )

        dense = planner.build_layer(block, 0, cluster.af.links[0])
        dense_stages = Counter(node.stage for node in dense.nodes)
        self.assertFalse(
            {"router", "routed_expert", "shared_expert", "weighted_combine"}
            & set(dense_stages)
        )
        self.assertEqual(dense_stages["dense_ffn"], 4)
        self.assertEqual(dense_stages["residual"], 2)
        self.assertEqual(dense_stages["encode"], 10)
        self.assertEqual(dense_stages["decode"], 16)
        dense_collectives = [
            node.operation
            for node in dense.nodes
            if node.kind == "communication"
            and node.operation in {"all_reduce", "reduce_scatter", "all_gather"}
        ]
        self.assertEqual(
            dense_collectives, ["all_reduce", "reduce_scatter", "all_gather"]
        )
        dense_a_to_f = [
            node for node in dense.nodes if node.operation == "af_a_to_f"
        ]
        self.assertEqual(len(dense_a_to_f), 8)
        self.assertEqual(
            {node.source_endpoint.domain_rank for node in dense_a_to_f}, {0}
        )
        self.assertEqual(
            {node.target_endpoint.domain_rank for node in dense_a_to_f},
            {0, 1, 2, 3},
        )
        self.assertTrue(all(node.payload_bytes == 16 for node in dense_a_to_f))
        dense_f_to_a = [
            node for node in dense.nodes if node.operation == "af_f_to_a"
        ]
        self.assertEqual(len(dense_f_to_a), 8)
        self.assertEqual(
            {
                (node.source_endpoint.domain_rank, node.target_endpoint.domain_rank)
                for node in dense_f_to_a
            },
            {(0, 0), (1, 0), (2, 1), (3, 1)},
        )
        self.assertEqual(
            dense.terminal_node_ids, ("i5:b0:l0:residual:layout",)
        )

        moe = planner.build_layer(block, 1, cluster.af.links[0])
        moe_stages = Counter(node.stage for node in moe.nodes)
        self.assertEqual(moe_stages["router"], 1)
        self.assertEqual(moe_stages["routed_expert"], 2)
        self.assertEqual(moe_stages["shared_expert"], 4)
        self.assertEqual(moe_stages["weighted_combine"], 2)
        self.assertEqual(moe_stages["residual"], 2)
        self.assertEqual(moe_stages["encode"], 18)
        self.assertEqual(moe_stages["decode"], 24)
        routed = [node for node in moe.nodes if node.stage == "routed_expert"]
        self.assertEqual(
            sum(node.local_work.assignment_count for node in routed), 4
        )
        self.assertEqual(
            {node.local_work.expert_workload for node in routed},
            {((0, 2),), ((1, 2),)},
        )
        shared = [node for node in moe.nodes if node.stage == "shared_expert"]
        self.assertTrue(
            all(node.local_work.original_token_count == 2 for node in shared)
        )
        moe_a_to_f = [node for node in moe.nodes if node.operation == "af_a_to_f"]
        self.assertEqual(len(moe_a_to_f), 8)
        self.assertEqual(
            len(
                {
                    (
                        node.source_endpoint.domain_rank,
                        node.target_endpoint.domain_rank,
                    )
                    for node in moe_a_to_f
                }
            ),
            4,
        )
        self.assertTrue(all(node.payload_bytes == 16 for node in moe_a_to_f))
        moe_f_to_a = [node for node in moe.nodes if node.operation == "af_f_to_a"]
        self.assertEqual(len(moe_f_to_a), 16)
        owner_intervals = {0: (0, 8), 1: (8, 16)}
        for node in moe_f_to_a:
            owner_interval = owner_intervals[node.target_endpoint.domain_rank]
            self.assertTrue(
                all(
                    owner_interval[0] <= tensor_slice.interval[0]
                    and tensor_slice.interval[1] <= owner_interval[1]
                    for tensor_slice in node.tensor_slices
                )
            )
        routed_slices = [
            tensor_slice
            for node in moe_f_to_a
            for tensor_slice in node.tensor_slices
            if ":routed:" in tensor_slice.tensor_item_id
        ]
        self.assertEqual(len(routed_slices), 8)
        self.assertEqual(
            {tensor_slice.routing_weight for tensor_slice in routed_slices},
            {0.75, 0.25},
        )
        self.assertTrue(all(tensor_slice.partial_id for tensor_slice in routed_slices))
        combine = [node for node in moe.nodes if node.stage == "weighted_combine"]
        for node in combine:
            self.assertEqual(node.local_work.original_token_count, 2)
            self.assertEqual(node.local_work.assignment_count, 4)
            self.assertEqual(node.local_work.input_edge_count, 6)
            self.assertEqual(node.local_work.output_token_count, 2)
            self.assertEqual(dict(node.local_work.tensor_shape)["hidden_size"], 8)
            self.assertEqual(
                {edge.routing_weight for edge in node.local_work.routing_edges},
                {0.75, 0.25},
            )
        self.assertEqual(
            moe.terminal_node_ids, ("i5:b0:l1:residual:layout",)
        )

    def test_topology_matrix_covers_single_tp_dp_ep_and_combined_modes(self):
        cases = (
            ("single", 1, 1, 1, 1, 1),
            ("tp-only", 2, 1, 2, 1, 1),
            ("dp-only", 1, 2, 1, 2, 1),
            ("rank-reuse", 1, 1, 1, 2, 2),
            ("combined", 2, 2, 2, 2, 4),
        )
        for name, a_tp, a_dp, f_tp, f_dp, f_ep in cases:
            with self.subTest(name=name):
                cluster = _cluster(a_tp, a_dp, f_tp, f_dp, f_ep)
                planner = ParallelExecutionPlanner(
                    cluster,
                    hidden_size=16,
                    dtype_size_bytes=2,
                    num_experts=4,
                    top_k=1,
                )
                a_domain = cluster.execution_domains["a-domain"]
                f_domain = cluster.execution_domains["f-domain"]
                placement = planner.placement(
                    "decode", "a-pool:0", f_domain.logical_instances[0]
                )
                block = _block()
                block = block.__class__(
                    **{
                        **block.__dict__,
                        "source_a_worker_id": 0,
                        "placement": placement,
                    }
                )
                dag = planner.build_layer(block, 0, cluster.af.links[0])
                self.assertEqual(len(a_domain.logical_instances), a_dp)
                self.assertEqual(len(f_domain.logical_instances), f_dp)
                self.assertEqual(
                    sum(node.operation == "a_rank_compute" for node in dag.nodes),
                    a_tp,
                )
                active_ep = 2 if f_ep > 1 else 1
                self.assertEqual(
                    sum(node.operation == "expert_rank_compute" for node in dag.nodes),
                    active_ep if f_ep > 1 else f_tp,
                )
                expected_collectives = int(a_tp > 1) + int(
                    f_ep == 1 and f_tp > 1
                )
                self.assertEqual(
                    sum(node.operation == "tp_collective" for node in dag.nodes),
                    expected_collectives,
                )

    def test_global_round_robin_is_scoped_by_phase_and_domain(self):
        cluster = _cluster()
        selector = GlobalRoundRobinReplicaSelector()
        a_domain = cluster.execution_domains["a-domain"]
        self.assertEqual(
            [selector.select("decode", a_domain).name for _ in range(4)],
            ["a0", "a1", "a0", "a1"],
        )
        self.assertEqual(selector.select("prefill", a_domain).name, "a0")

    def test_combined_tp_dp_ep_graph_uses_local_work_and_typed_communication(self):
        cluster = _cluster()
        planner = ParallelExecutionPlanner(
            cluster, hidden_size=16, dtype_size_bytes=2, num_experts=4, top_k=1
        )
        block = _block()
        block = block.__class__(
            **{
                **block.__dict__,
                "placement": planner.placement(
                    "decode",
                    "a-pool:2",
                    cluster.execution_domains["f-domain"].logical_instances[0],
                ),
            }
        )
        self.assertEqual(
            [
                (item.expert_id, item.ep_rank)
                for item in block.placement.expert_placement
            ],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )
        self.assertTrue(
            all(
                len(item.domain_ranks) == 1 for item in block.placement.expert_placement
            )
        )
        dag = planner.build_layer(block, 0, cluster.af.links[0])
        operations = [node.operation for node in dag.nodes]

        self.assertEqual(operations.count("af_a_to_f"), 1)
        self.assertEqual(operations.count("af_f_to_a"), 1)
        self.assertEqual(operations.count("ep_dispatch"), 0)
        self.assertEqual(operations.count("ep_combine"), 0)
        self.assertEqual(operations.count("tp_collective"), 1)

        a_nodes = [node for node in dag.nodes if node.operation == "a_rank_compute"]
        self.assertEqual([node.dp_rank for node in a_nodes], [1, 1])
        self.assertEqual(
            [dict(node.local_work.tensor_shape)["hidden_size"] for node in a_nodes],
            [8, 8],
        )
        self.assertTrue(
            all(
                node.physical_workers[0].startswith("a-pool:2")
                or node.physical_workers[0].startswith("a-pool:3")
                for node in a_nodes
            )
        )

        router = next(node for node in dag.nodes if node.operation == "router_topk")
        a_collective = next(
            node
            for node in dag.nodes
            if node.operation == "tp_collective" and node.source_domain == "a-domain"
        )
        a_to_f = next(node for node in dag.nodes if node.operation == "af_a_to_f")
        self.assertEqual(router.predecessor_ids, (a_collective.node_id,))
        self.assertEqual(router.source_domain, "a-domain")
        self.assertEqual(router.physical_workers, ("a-pool:2",))
        self.assertEqual(router.tp_rank, 0)
        self.assertEqual(router.ep_rank, 0)
        self.assertEqual(dict(router.local_work.tensor_shape)["hidden_size"], 16)
        self.assertEqual(a_to_f.predecessor_ids, (router.node_id,))
        self.assertEqual(
            set(a_to_f.physical_workers),
            {"a-pool:2", "f-pool:0", "f-pool:2"},
        )

        experts = [
            node for node in dag.nodes if node.operation == "expert_rank_compute"
        ]
        self.assertEqual(
            {(node.ep_rank, node.local_work.token_count) for node in experts},
            {(0, 10), (2, 2)},
        )
        self.assertEqual(
            {node.ep_rank: dict(node.local_work.expert_workload) for node in experts},
            {0: {0: 10}, 2: {2: 2}},
        )
        self.assertEqual({node.dp_rank for node in experts}, {0, 1})
        expert_joins = {
            node.node_id
            for node in dag.nodes
            if node.operation == "expert_rank_compute"
        }
        f_to_a = next(node for node in dag.nodes if node.operation == "af_f_to_a")
        self.assertEqual(a_to_f.payload_bytes, 12 * 16 * 2)
        self.assertEqual(f_to_a.payload_bytes, 12 * 16 * 2)
        self.assertEqual(set(f_to_a.predecessor_ids), expert_joins)
        self.assertEqual(set(f_to_a.physical_workers), set(a_to_f.physical_workers))

    def test_a_to_f_payload_uses_routed_token_count(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        planner = ParallelExecutionPlanner(
            cluster, hidden_size=16, dtype_size_bytes=2, num_experts=4, top_k=2
        )
        block = _block(histogram=((0, 12), (2, 12)))
        block = block.__class__(
            **{
                **block.__dict__,
                "source_a_worker_id": 0,
                "placement": planner.placement(
                    "decode",
                    "a-pool:0",
                    cluster.execution_domains["f-domain"].logical_instances[0],
                ),
            }
        )

        dag = planner.build_layer(block, 0, cluster.af.links[0])
        a_to_f = next(node for node in dag.nodes if node.operation == "af_a_to_f")
        f_to_a = next(node for node in dag.nodes if node.operation == "af_f_to_a")

        self.assertEqual(a_to_f.payload_bytes, 24 * 16 * 2)
        self.assertEqual(f_to_a.payload_bytes, 24 * 16 * 2)

    def test_ep_one_uses_only_cross_domain_dispatch_and_combine(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        planner = ParallelExecutionPlanner(
            cluster, hidden_size=16, dtype_size_bytes=2, num_experts=4, top_k=1
        )
        block = _block(histogram=((0, 12),))
        block = block.__class__(
            **{
                **block.__dict__,
                "source_a_worker_id": 0,
                "placement": planner.placement(
                    "decode",
                    "a-pool:0",
                    cluster.execution_domains["f-domain"].logical_instances[0],
                ),
            }
        )

        dag = planner.build_layer(block, 0, cluster.af.links[0])
        operations = {node.operation for node in dag.nodes}
        router = next(node for node in dag.nodes if node.operation == "router_topk")
        a_to_f = next(node for node in dag.nodes if node.operation == "af_a_to_f")
        expert = next(
            node for node in dag.nodes if node.operation == "expert_rank_compute"
        )
        f_to_a = next(node for node in dag.nodes if node.operation == "af_f_to_a")

        self.assertNotIn("ep_dispatch", operations)
        self.assertNotIn("ep_combine", operations)
        self.assertEqual(a_to_f.predecessor_ids, (router.node_id,))
        self.assertEqual(expert.predecessor_ids, (a_to_f.node_id,))
        self.assertEqual(f_to_a.predecessor_ids, (expert.node_id,))

    def test_empty_ep_rank_has_no_compute_and_tp_one_has_no_collective(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=2, f_ep=2)
        planner = ParallelExecutionPlanner(
            cluster, hidden_size=16, dtype_size_bytes=2, num_experts=4, top_k=1
        )
        block = _block(histogram=((0, 12),))
        block = block.__class__(
            **{
                **block.__dict__,
                "source_a_worker_id": 0,
                "placement": AFBlockPlacement("a-domain", "a0", 0, "f-domain", "f0", 0),
            }
        )
        dag = planner.build_layer(block, 0, cluster.af.links[0])
        experts = [
            node for node in dag.nodes if node.operation == "expert_rank_compute"
        ]
        self.assertEqual(len(experts), 1)
        self.assertEqual(experts[0].ep_rank, 0)
        self.assertNotIn("tp_collective", {node.operation for node in dag.nodes})

    def test_round_robin_placement_and_incomplete_snapshot_are_validated(self):
        cluster = _cluster(expert_placement_strategy="round_robin")
        planner = ParallelExecutionPlanner(
            cluster, hidden_size=16, dtype_size_bytes=2, num_experts=4, top_k=1
        )
        placement = planner.placement(
            "decode",
            "a-pool:0",
            cluster.execution_domains["f-domain"].logical_instances[1],
        )
        self.assertEqual(
            [(item.expert_id, item.ep_rank) for item in placement.expert_placement],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )
        block = _block(histogram=((3, 1),))
        incomplete = placement.__class__(
            **{
                **placement.__dict__,
                "expert_placement": placement.expert_placement[:-1],
            }
        )
        block = block.__class__(**{**block.__dict__, "placement": incomplete})
        with self.assertRaisesRegex(
            ConfigurationError, "expert id 3.*F domain.*DP replica"
        ):
            planner.build_layer(block, 0, cluster.af.links[0])


if __name__ == "__main__":
    unittest.main()
