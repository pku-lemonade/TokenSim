from __future__ import annotations

from types import SimpleNamespace
import unittest

from TokenSim.af.planning.batches import AFBlock
from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.providers.queries import AFEndpoint, AFTensorSlice
from TokenSim.errors import ConfigurationError
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment


def _request(request_id: int, phase: str):
    is_prefill = phase == "prefill"
    return SimpleNamespace(
        id=request_id,
        is_prefill=is_prefill,
        prefill_len=1,
        prefill_compute_len=1,
        context_len=1 if is_prefill else 9,
    )


def _snapshot(
    iteration_id: int,
    phase: str,
    layer_id: int,
    requests: tuple[SimpleNamespace, ...],
    decode_step: int | None = None,
) -> LayerRoutingSnapshot:
    resolved_step = (
        decode_step if decode_step is not None else (0 if phase == "prefill" else 1)
    )
    original_tokens = tuple(
        (request.id, 0 if phase == "prefill" else request.context_len - 1)
        for request in requests
    )
    assignments = tuple(
        RoutingAssignment(
            iteration_id=iteration_id,
            phase=phase,
            decode_step=resolved_step,
            layer_id=layer_id,
            request_id=request_id,
            token_id=token_id,
            top_k_index=0,
            expert_id=index,
            routing_weight=1.0,
            a_dp_rank=0,
            a_tp_rank=0,
        )
        for index, (request_id, token_id) in enumerate(original_tokens)
    )
    return LayerRoutingSnapshot(
        iteration_id=iteration_id,
        phase=phase,
        decode_step=resolved_step,
        layer_id=layer_id,
        top_k=1,
        original_tokens=original_tokens,
        assignments=assignments,
    )


class AFBlockRoutingSnapshotTest(unittest.TestCase):
    def test_builder_freezes_only_moe_layer_snapshots(self):
        requests = tuple(_request(request_id, "decode") for request_id in (10, 11))
        calls: list[int] = []

        def factory(iteration_id, phase, layer_id, block_requests, _placement):
            calls.append(layer_id)
            return _snapshot(iteration_id, phase, layer_id, block_requests)

        iteration = AFBlockBuilder().build(
            iteration_id=7,
            requests=requests,
            af_block_size=2,
            source_a_worker_id=0,
            link=SimpleNamespace(name="af-link", phases=("decode",)),
            layer_types={0: "dense", 1: "moe", 2: "dense", 3: "moe"},
            routing_snapshot_factory=factory,
        )

        block = iteration.blocks[0]
        self.assertEqual(calls, [1, 3])
        self.assertEqual(block.routing_layer_ids, (1, 3))
        self.assertIsNone(block.routing_snapshot_for_layer(0))
        self.assertIsNot(
            block.routing_snapshot_for_layer(1),
            block.routing_snapshot_for_layer(3),
        )

    def test_snapshot_phase_isolated_and_mismatch_rejected(self):
        def factory(iteration_id, phase, layer_id, block_requests, _placement):
            return _snapshot(iteration_id, phase, layer_id, block_requests)

        builder = AFBlockBuilder()
        prefill = builder.build(
            1,
            (_request(1, "prefill"),),
            1,
            0,
            SimpleNamespace(name="prefill-link", phases=("prefill",)),
            layer_types={0: "moe"},
            routing_snapshot_factory=factory,
        ).blocks[0]
        decode = builder.build(
            2,
            (_request(1, "decode"),),
            1,
            0,
            SimpleNamespace(name="decode-link", phases=("decode",)),
            layer_types={0: "moe"},
            routing_snapshot_factory=factory,
        ).blocks[0]

        self.assertEqual(prefill.routing_snapshot_for_layer(0).phase, "prefill")
        self.assertEqual(decode.routing_snapshot_for_layer(0).phase, "decode")
        self.assertIsNot(
            prefill.routing_snapshot_for_layer(0),
            decode.routing_snapshot_for_layer(0),
        )

        bad_snapshot = _snapshot(1, "decode", 0, (_request(1, "decode"),))
        with self.assertRaisesRegex(ConfigurationError, "phase mismatch"):
            AFBlock(
                iteration_id=1,
                block_id=0,
                phase="prefill",
                request_ids=(1,),
                request_count=1,
                token_count=1,
                context_lengths=(1,),
                source_a_worker_id=0,
                link_name="af-link",
                routing_snapshots=(bad_snapshot,),
            )

        with self.assertRaisesRegex(ConfigurationError, "multiple steps"):
            AFBlock(
                iteration_id=2,
                block_id=0,
                phase="decode",
                request_ids=(1,),
                request_count=1,
                token_count=1,
                context_lengths=(9,),
                source_a_worker_id=0,
                link_name="af-link",
                routing_snapshots=(
                    _snapshot(2, "decode", 0, (_request(1, "decode"),), 1),
                    _snapshot(2, "decode", 1, (_request(1, "decode"),), 2),
                ),
            )


class AFExecutionDAGModelTest(unittest.TestCase):
    def _node(self, node_id: str, predecessors=(), stage="attention"):
        return ExecutionNode(
            node_id=node_id,
            operation=stage,
            kind="compute",
            predecessor_ids=predecessors,
            stage=stage,
            direction=None,
            source_domain="a-domain",
            target_domain="a-domain",
            logical_participants=("a0",),
            domain_ranks=(0,),
            physical_workers=("a-pool:0",),
            dp_rank=0,
            local_work=RankLocalWork(token_count=1, original_token_count=1),
        )

    def _sinks(self):
        root = self._node("root")
        left = self._node("left", ("root",))
        right = self._node("right", ("root",))
        return root, left, right

    def test_dag_infers_all_sinks_and_single_terminal_compatibility(self):
        root = self._node("root")
        terminal = self._node("terminal", ("root",), stage="residual")
        dag = AFExecutionDAG((root, terminal))

        self.assertEqual(dag.terminal_node_ids, ("terminal",))
        self.assertEqual(dag.final_node_id, "terminal")

        root, left, right = self._sinks()
        multi = AFExecutionDAG((root, left, right))
        self.assertEqual(multi.terminal_node_ids, ("left", "right"))
        with self.assertRaisesRegex(ConfigurationError, "multiple terminal"):
            _ = multi.final_node_id

    def test_all_execution_stages_are_explicitly_representable(self):
        stages = (
            "attention",
            "router",
            "dense_ffn",
            "routed_expert",
            "shared_expert",
            "weighted_combine",
            "residual",
            "cast",
            "layout",
        )
        self.assertEqual(
            tuple(self._node(stage, stage=stage).stage for stage in stages),
            stages,
        )

    def test_dag_rejects_invalid_graph_and_terminal_sets(self):
        node = self._node("root")
        with self.assertRaisesRegex(ConfigurationError, "duplicate node"):
            AFExecutionDAG((node, node))
        with self.assertRaisesRegex(ConfigurationError, "missing or cyclic"):
            AFExecutionDAG((self._node("child", ("absent",)),))
        with self.assertRaisesRegex(ConfigurationError, "out-of-order or cyclic"):
            AFExecutionDAG((self._node("child", ("root",)), node))

        root, left, right = self._sinks()
        with self.assertRaisesRegex(ConfigurationError, "do not exist"):
            AFExecutionDAG((root, left, right), terminal_node_ids=("left", "absent"))
        with self.assertRaisesRegex(ConfigurationError, "equal all sink"):
            AFExecutionDAG((root, left, right), terminal_node_ids=("left",))

    def test_work_and_node_retain_owner_layout_and_stage_contract(self):
        endpoint = AFEndpoint(
            domain="a-domain",
            domain_rank=1,
            physical_worker="a-pool:1",
            device_type="test-device",
            dp_rank=0,
            tp_rank=1,
        )
        tensor_slice = AFTensorSlice(
            tensor_item_id="request:1:token:8:routed:0",
            interval=(8, 16),
            layout="partial",
            owner_domain="a-domain",
            owner_domain_rank=1,
            owner_dp_rank=0,
            owner_tp_rank=1,
            owner_physical_worker="a-pool:1",
            partial_id="expert:3",
        )
        work = RankLocalWork(
            token_count=2,
            original_token_count=1,
            assignment_count=2,
            input_edge_count=2,
            output_token_count=1,
            tensor_slices=(tensor_slice,),
            owner_endpoint=endpoint,
            source_layout="partial",
            target_layout="sharded",
        )
        node = ExecutionNode(
            node_id="combine",
            operation="weighted_combine",
            kind="compute",
            predecessor_ids=(),
            stage="weighted_combine",
            direction=None,
            source_domain="a-domain",
            target_domain="a-domain",
            logical_participants=("a0",),
            domain_ranks=(1,),
            physical_workers=("a-pool:1",),
            dp_rank=0,
            tp_rank=1,
            local_work=work,
            layer_type="moe",
            logical_stage_id="i1:b0:l2:combine",
            source_endpoint=endpoint,
            target_endpoint=endpoint,
            tensor_slices=(tensor_slice,),
            source_layout="partial",
            target_layout="sharded",
        )

        self.assertEqual(node.stage, "weighted_combine")
        self.assertEqual(work.as_dict()["assignment_count"], 2)
        self.assertEqual(
            work.as_dict()["owner_endpoint"]["physical_worker"], "a-pool:1"
        )


if __name__ == "__main__":
    unittest.main()
