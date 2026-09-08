import unittest

from TokenSim.af.models import AFBlock, AFBlockPlacement
from TokenSim.af.planning import (
    AFExecutionDAG,
    ExecutionNode,
    GlobalRoundRobinReplicaSelector,
    ParallelExecutionPlanner,
    RankLocalWork,
)
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError


def _replica(group, name, dp_rank, tp_size, ep_size, worker_offset):
    ranks_per_replica = tp_size * ep_size
    return {
        "name": name,
        "rank_mapping": [
            {
                "domain_rank": dp_rank * ranks_per_replica + local_rank,
                "tp_rank": local_rank % tp_size,
                "pp_rank": 0,
                "dp_rank": dp_rank,
                "ep_rank": local_rank // tp_size,
                "physical_worker": {
                    "group_name": group,
                    "worker_index": worker_offset + local_rank,
                },
            }
            for local_rank in range(ranks_per_replica)
        ],
    }


def _cluster(
    a_tp=2,
    a_dp=2,
    f_tp=2,
    f_dp=2,
    f_ep=2,
    expert_placement_strategy="linear",
):
    a_workers = a_tp * a_dp
    f_workers = f_tp * f_dp * f_ep
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
                            rank * f_tp * f_ep,
                        )
                        for rank in range(f_dp)
                    ],
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


class ParallelExecutionPlannerTest(unittest.TestCase):
    def test_topology_matrix_covers_single_tp_dp_ep_and_combined_modes(self):
        cases = (
            ("single", 1, 1, 1, 1, 1),
            ("tp-only", 2, 1, 2, 1, 1),
            ("dp-only", 1, 2, 1, 2, 1),
            ("ep-only", 1, 1, 1, 1, 2),
            ("combined", 2, 2, 2, 2, 2),
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
                    active_ep * f_tp,
                )
                expected_collectives = int(a_tp > 1) + active_ep * int(f_tp > 1)
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
            [(0, 0), (1, 0), (2, 1), (3, 1)],
        )
        self.assertTrue(
            all(
                len(item.domain_ranks) == 2 for item in block.placement.expert_placement
            )
        )
        dag = planner.build_layer(block, 0, cluster.af.links[0])
        operations = [node.operation for node in dag.nodes]

        self.assertEqual(operations.count("af_a_to_f"), 1)
        self.assertEqual(operations.count("af_f_to_a"), 1)
        self.assertEqual(operations.count("ep_dispatch"), 0)
        self.assertEqual(operations.count("ep_combine"), 0)
        self.assertEqual(operations.count("tp_collective"), 3)

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
            {"a-pool:2", "f-pool:0", "f-pool:1", "f-pool:2", "f-pool:3"},
        )

        experts = [
            node for node in dag.nodes if node.operation == "expert_rank_compute"
        ]
        self.assertEqual(
            {(node.ep_rank, node.local_work.token_count) for node in experts},
            {(0, 10), (1, 2)},
        )
        self.assertEqual(
            {node.ep_rank: dict(node.local_work.expert_workload) for node in experts},
            {0: {0: 10}, 1: {2: 2}},
        )
        self.assertTrue(all(node.dp_rank == 0 for node in experts))
        expert_joins = {
            node.node_id
            for node in dag.nodes
            if node.operation == "tp_collective" and node.source_domain == "f-domain"
        }
        f_to_a = next(node for node in dag.nodes if node.operation == "af_f_to_a")
        self.assertEqual(a_to_f.payload_bytes, 12 * 16 * 2)
        self.assertEqual(f_to_a.payload_bytes, 12 * 16 * 2)
        self.assertEqual(set(f_to_a.predecessor_ids), expert_joins)
        self.assertEqual(set(f_to_a.physical_workers), set(a_to_f.physical_workers))

    def test_a_to_f_payload_uses_routed_token_count(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=2)
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
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=2)
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
            [(0, 0), (1, 1), (2, 0), (3, 1)],
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
