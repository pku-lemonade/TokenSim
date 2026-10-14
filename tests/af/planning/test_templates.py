from dataclasses import replace
import unittest
from unittest.mock import patch

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.planning.batches import AFBlock
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.routing import LayerRoutingSnapshot, RoutingAssignment
from scripts.deepseek_v3_runtime import configuration, simulate
from tests.af.support.parallel import _cluster
from tests.af.support.planning import _formal_runtime_plan
from tests.af.support.services import observe_services


class TemplateTest(unittest.TestCase):
    def test_replica_placement_and_tp_ep_are_part_of_template_identity(self):
        for tp in (1, 2):
            cluster = _cluster(a_tp=tp, a_dp=2, f_tp=2, f_dp=2, f_ep=4)
            moe = MoEModelConfig(
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
            plan = _formal_runtime_plan(cluster, moe)
            planner = ParallelExecutionPlanner(cluster, runtime_plan=plan)
            for replica in (0, 1, 0):
                placement = planner.placement(
                    "decode",
                    f"a-pool:{replica*tp}",
                    cluster.execution_domains["f-domain"].logical_instances[replica],
                )
                assignment = RoutingAssignment(
                    iteration_id=0,
                    phase="decode",
                    decode_step=0,
                    layer_id=0,
                    request_id=0,
                    token_id=7,
                    top_k_index=0,
                    expert_id=0,
                    routing_weight=1.0,
                    a_dp_rank=replica,
                    a_tp_rank=0,
                )
                snapshot = LayerRoutingSnapshot(
                    iteration_id=0,
                    phase="decode",
                    decode_step=0,
                    layer_id=0,
                    top_k=1,
                    original_tokens=((0, 7),),
                    assignments=(assignment,),
                )
                block = AFBlock(
                    0,
                    0,
                    "decode",
                    (0,),
                    1,
                    1,
                    (8,),
                    replica * tp,
                    "af-link",
                    placement,
                    routing_snapshots=(snapshot,),
                )
                bound = planner.build_layer(block, 0, cluster.af.links[0])
                reference = planner._build_runtime_layer(block, 0, cluster.af.links[0])
                self.assertEqual(bound.nodes, reference.nodes)
            self.assertEqual(planner.template_cache.hits, 1)
            self.assertEqual(len(planner.template_cache.templates), 2)

    def test_cached_dags_match_reference_for_every_bound_field(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=2)
        original = ParallelExecutionPlanner.build_layer
        compared = []

        def checked(planner, block, layer, link):
            bound = original(planner, block, layer, link)
            reference = planner._build_runtime_layer(block, layer, link)
            self.assertEqual(bound.nodes, reference.nodes)
            self.assertEqual(bound.terminal_node_ids, reference.terminal_node_ids)
            # Validate each test binding independently, including token ownership.
            AFExecutionDAG(bound.nodes)
            again = original(planner, block, layer, link)
            self.assertEqual(again.nodes, reference.nodes)
            compared.extend((bound, again))
            return bound

        with patch.object(ParallelExecutionPlanner, "build_layer", checked):
            executor, _, _, _ = simulate(
                config, 4, 8, steps=3, fallback="analytical_only"
            )
        cache = executor.plan_builder.template_cache
        self.assertGreater(cache.hits, 0)
        self.assertEqual(cache.requests, len(compared))
        self.assertIs(compared[0].predecessor_indices, compared[1].predecessor_indices)
        self.assertIs(
            compared[0].nodes[0].physical_workers,
            compared[1].nodes[0].physical_workers,
        )
        self.assertIs(compared[0].nodes[0].local_work, compared[1].nodes[0].local_work)
        with self.assertRaises(TypeError):
            compared[0].nodes[0].metadata["changed"] = True

    def test_service_trace_and_completion_differential_matrix(self):
        for tp, block, batch, context, router, dtype in (
            (1, 1, 2, 8, "a", "bf16"),
            (2, 2, 4, 16, "f", "bf16"),
            (2, 1, 2, 9, "a", "fp8"),
        ):
            with self.subTest(tp=tp, block=block, dtype=dtype):
                results = []
                for enabled in (False, True):
                    config = configuration(a_tp=tp, f_tp=2, af_batch_size=block)
                    config.doc["af"].update(
                        dag_template_cache=enabled,
                        service_estimate_cache=enabled,
                        router_location=router,
                        kv_cache_dtype=dtype,
                        communication_timing=dict(
                            enabled=True, host_submit_us=5, host_completion_us=2
                        ),
                    )
                    config = build_model_runtime(
                        config.doc, hardware=config.hardware, model=config.model
                    )
                    with observe_services() as services:
                        executor, rows, _, _ = simulate(
                            config, batch, context, steps=2, fallback="analytical_only"
                        )
                    results.append((rows, services, executor.trace.events))
                self.assertEqual(results[0], results[1])

    def test_routing_changes_miss_and_eviction_is_bounded(self):
        config = configuration(af_batch_size=1, f_tp=2)
        executor, _, _, _ = simulate(config, 1, 8, fallback="analytical_only")
        planner = executor.plan_builder
        cache = planner.template_cache
        # Capture a valid block; completed iterations are normally released.
        iteration = executor.submit(
            0,
            [
                type(
                    "Request",
                    (),
                    dict(
                        id=7,
                        is_prefill=False,
                        prefill_len=8,
                        context_len=8,
                        generation_idx=1,
                    ),
                )()
            ],
            "decode",
        )
        block = iteration.blocks[0]
        snapshot = block.routing_snapshot_for_layer(0)
        link = config.cluster.af.links[0]
        cache.limit = 2
        for shift in (0, 64, 128):
            assignments = tuple(
                replace(a, expert_id=(a.expert_id + shift) % 256)
                for a in snapshot.assignments
            )
            changed = replace(
                block, routing_snapshots=(replace(snapshot, assignments=assignments),)
            )
            bound = planner.build_layer(changed, 0, link)
            reference = planner._build_runtime_layer(changed, 0, link)
            self.assertEqual(bound.nodes, reference.nodes)
        self.assertEqual(len(cache.templates), 2)
