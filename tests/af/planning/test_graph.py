"""Af parallel planning."""

from collections import Counter
from dataclasses import replace
import unittest

from TokenSim.af.planning.batches import AFBlock
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.errors import ConfigurationError
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment
from tests.af.support.parallel import _cluster
from tests.af.support.planning import _formal_runtime_plan, _small_formal_moe_dag


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
        with self.assertRaisesRegex(
            ConfigurationError, "duplicate physical tensor copy"
        ):
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
        planner = ParallelExecutionPlanner(cluster, runtime_plan=runtime_plan)
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
        dense_a_to_f = [node for node in dense.nodes if node.operation == "af_a_to_f"]
        self.assertEqual(len(dense_a_to_f), 8)
        self.assertEqual(
            {node.source_endpoint.domain_rank for node in dense_a_to_f}, {0}
        )
        self.assertEqual(
            {node.target_endpoint.domain_rank for node in dense_a_to_f},
            {0, 1, 2, 3},
        )
        self.assertTrue(all(node.payload_bytes == 16 for node in dense_a_to_f))
        dense_f_to_a = [node for node in dense.nodes if node.operation == "af_f_to_a"]
        self.assertEqual(len(dense_f_to_a), 8)
        self.assertEqual(
            {
                (node.source_endpoint.domain_rank, node.target_endpoint.domain_rank)
                for node in dense_f_to_a
            },
            {(0, 0), (1, 0), (2, 1), (3, 1)},
        )
        self.assertIn("i5:b0:l0:residual:layout", dense.terminal_node_ids)
        self.assertIn("i5:b0:l0:post_attention_norm:r0", dense.terminal_node_ids)
        self.assertIn("i5:b0:l0:post_attention_norm:r1", dense.terminal_node_ids)

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
        self.assertEqual(sum(node.local_work.assignment_count for node in routed), 4)
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
        self.assertEqual(moe.terminal_node_ids, ("i5:b0:l1:residual:layout",))
