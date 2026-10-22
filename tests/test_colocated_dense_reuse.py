"""Numerical tensor flow and native execution checks for colocated dense reuse."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.af.serving.service import AFService
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration
from scripts.lpx.replay_profiles import cached_profile


def tensor_oracle(dag, seed=123):
    """Execute planned slices with independent dense matrices and RMS normalization."""
    attention = [node for node in dag.nodes if node.stage == "attention"]
    work = attention[0].local_work
    shape = dict(work.tensor_shape)
    count, hidden, tp = work.token_count, shape["global_hidden_size"], len(attention)
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(count, hidden))
    attention_weights = rng.normal(size=(tp, hidden, hidden))
    up, down = rng.normal(size=(hidden, 64)), rng.normal(size=(64, hidden))
    token_ids = sorted(
        {
            item.tensor_item_id.rsplit(":", 1)[0]
            for node in dag.nodes
            for item in node.tensor_slices
        }
    )
    rows = {token: i for i, token in enumerate(token_ids)}
    values, residual = {}, None
    for node in dag.nodes:
        parents = [values[item] for item in node.predecessor_ids]
        if node.stage == "attention":
            value = inputs @ attention_weights[node.tp_rank]
        elif node.operation == "all_reduce":
            value = sum(parents)
        elif node.operation == "post_attention_norm":
            value = parents[0] + inputs
            value = value / np.sqrt(np.mean(value**2, axis=-1, keepdims=True) + 1e-6)
            residual = value
        elif node.operation in {"af_a_to_f", "af_f_to_a"}:
            value = np.full_like(inputs, np.nan)
            for item in node.tensor_slices:
                row = rows[item.tensor_item_id.rsplit(":", 1)[0]]
                start, end = item.interval
                value[row, start:end] = parents[0][row, start:end]
        elif node.stage == "dense_ffn":
            activation = np.full_like(inputs, np.nan)
            for parent in parents:
                activation = np.where(np.isnan(parent), activation, parent)
            if np.isnan(activation).any():
                raise AssertionError("FFN input has missing tensor elements")
            start, end = node.tp_rank * 64 // tp, (node.tp_rank + 1) * 64 // tp
            value = np.tanh(activation @ up[:, start:end]) @ down[start:end]
        elif node.stage == "residual":
            value = parents[0] + residual
        elif node.operation == "all_gather":
            value = np.full_like(inputs, np.nan)
            for parent in parents:
                if np.any(~np.isnan(value) & ~np.isnan(parent)):
                    raise AssertionError("gather duplicates tensor ownership")
                value = np.where(np.isnan(parent), value, parent)
        else:
            raise AssertionError(f"unexpected oracle node {node.operation}")
        values[node.node_id] = value
    by_id = {node.node_id: node for node in dag.nodes}
    return [
        values[node]
        for node in dag.terminal_node_ids
        if by_id[node].operation != "post_attention_norm"
    ]


class ColocatedDenseReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = replace(
            cls.hardware.model("gemma-4-31b-text"),
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            attention_windows=(),
            attention_geometries=(),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
        )

    def runtime(
        self, reuse=True, *, tp=2, pp=1, dp=1, edit=None, model=None, cache=False
    ):
        def build(doc, **kwargs):
            if edit:
                edit(doc)
            return build_model_runtime(doc, **kwargs)

        layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=tp,
            f_tp=tp,
            a_pp=pp,
            f_pp=pp,
            f_ep=1,
        )
        with patch("scripts.lpx.native_config.build_model_runtime", build):
            return configuration(
                self.hardware,
                model or self.model,
                layout,
                namespace="reuse",
                a_dp=dp,
                micro_batch=8,
                colocated_dense_reuse=reuse,
                dag_template_cache=cache,
            )

    def dag(self, runtime, *, layer=0, owner=0, f_replica=0, iteration=0, planner=None):
        planner = planner or ParallelExecutionPlanner(
            runtime.cluster, runtime_plan=runtime.plan
        )
        a, f = runtime.cluster.af_domains
        binding = runtime.plan.bindings_for_layer(
            a.name, a.logical_instances[owner].name, 0
        )[0]
        block = AFBlock(
            iteration_id=iteration,
            block_id=0,
            request_ids=(1 + iteration * 2, 2 + iteration * 2),
            request_count=2,
            token_count=3,
            context_lengths=(8, 10),
            source_a_worker_id=0,
            link_name=runtime.cluster.af_link.name,
            placement=planner.placement(
                binding.physical_identity, f.logical_instances[f_replica]
            ),
            query_lengths=(1, 2),
            query_offsets=(7, 8),
        )
        return planner.build_layer(block, layer, runtime.cluster.af_link)

    def execute(self, runtime, *, count=1):
        env = simpy.Environment()
        service = AFService(env, runtime, seed=42, retain_trace=True)
        sessions = [
            service.admit(i, 0, maximum_context=24, current_context=16)
            for i in range(count)
        ]
        for session in sessions:
            session.generated = 1
        env.run(until=env.process(service.forward(sessions, phase="decode")))
        return service

    def test_numeric_tensor_flow_matches_sharded_return_for_one_and_two_ranks(self):
        for tp in (1, 2):
            with self.subTest(tp=tp):
                reference = self.dag(self.runtime(False, tp=tp))
                local = self.dag(self.runtime(True, tp=tp))
                expected = tensor_oracle(reference)[0]
                for value in tensor_oracle(local):
                    np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
                self.assertFalse(
                    any(
                        node.kind == "communication"
                        and node.operation.startswith("af_")
                        for node in local.nodes
                    )
                )

    def test_physical_pairing_handles_permuted_logical_ffn_ranks(self):
        def permute(doc):
            mappings = doc["af"]["execution_domains"][1]["logical_instances"][0][
                "rank_mapping"
            ]
            mappings[0]["physical_worker"], mappings[1]["physical_worker"] = (
                mappings[1]["physical_worker"],
                mappings[0]["physical_worker"],
            )

        dag = self.dag(self.runtime(edit=permute))
        for node in dag.nodes:
            if node.metadata.get("colocated_dense_reuse"):
                self.assertTrue(
                    all(
                        item.owner_physical_worker == node.physical_workers[0]
                        for item in node.tensor_slices
                    )
                )
        for value in tensor_oracle(dag):
            np.testing.assert_allclose(value, tensor_oracle(dag)[0])

    def test_casts_sharded_collectives_and_remote_replicas_keep_original_protocol(self):
        edits = [
            lambda doc: doc["af"]["links"][0].update(a_to_f_dtype="fp8"),
            lambda doc: doc["af"]["links"][0].update(f_to_a_dtype="fp8"),
            lambda doc: doc["af"]["execution_domains"][1]["communication_plan"][
                0
            ].update(operation="reduce_scatter"),
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                self.assertEqual(
                    self.dag(self.runtime(False, edit=edit)),
                    self.dag(self.runtime(True, edit=edit)),
                )
        runtime = self.runtime(dp=2)
        dag = self.dag(runtime, owner=0, f_replica=1)
        self.assertTrue(any(node.operation == "af_a_to_f" for node in dag.nodes))
        local = self.dag(runtime, owner=1, f_replica=1)
        self.assertTrue(
            any(node.metadata.get("colocated_dense_reuse") for node in local.nodes)
        )

    def test_invalid_replica_ownership_coverage_and_missing_barriers_are_rejected(self):
        dag = self.dag(self.runtime())
        original = next(node for node in dag.nodes if node.stage == "residual")
        broken_slice = replace(
            original.tensor_slices[0], owner_physical_worker="reuse-a:999"
        )
        broken = replace(
            original, tensor_slices=(broken_slice, *original.tensor_slices[1:])
        )
        with self.assertRaisesRegex(ConfigurationError, "slice count|complete local"):
            AFExecutionDAG(
                tuple(broken if node == original else node for node in dag.nodes)
            )
        collective = next(
            node for node in dag.nodes if node.node_id.endswith("dense:collective")
        )
        with self.assertRaisesRegex(ConfigurationError, "collective input bindings"):
            broken = replace(
                collective,
                predecessor_ids=collective.predecessor_ids[:1],
                input_producer_ids=collective.input_producer_ids[:1],
            )
            AFExecutionDAG(
                tuple(broken if node == collective else node for node in dag.nodes)
            )
        broken = replace(
            collective, operation="reduce_scatter", target_layout="sharded"
        )
        with self.assertRaisesRegex(ConfigurationError, "complete AllReduce"):
            AFExecutionDAG(
                tuple(broken if node == collective else node for node in dag.nodes)
            )

    def test_native_costs_trace_validation_and_residual_memory(self):
        legacy, local = self.execute(self.runtime(False), count=2), self.execute(
            self.runtime(), count=2
        )
        self.assertLess(local.env.now, legacy.env.now)
        self.assertFalse(
            any(
                event.operation in {"af_a_to_f", "af_f_to_a"}
                for event in local.trace.events
            )
        )
        self.assertEqual(
            sum(event.operation == "all_reduce" for event in local.trace.events),
            sum(event.operation == "all_reduce" for event in legacy.trace.events),
        )
        for event in local.trace.events:
            if event.stage == "residual":
                self.assertEqual(event.local_work["tensor_shape"]["hidden_size"], 32)
                self.assertEqual(event.source_layout, "replicated")
        local.trace.validate_invariants(0)
        event = next(event for event in local.trace.events if event.stage == "residual")
        index = local.trace.events.index(event)
        local.trace.events[index] = replace(event, tensor_slices=())
        with self.assertRaisesRegex(SimulationStateError, "complete replicas"):
            local.trace.validate_invariants(0)
        for state in local.runtime.plan.physical_states:
            self.assertIn("colocated_dense_residual", dict(state.protocol_buffers))

    def test_pipeline_stages_template_rebinding_and_profile_cache(self):
        runtime = self.runtime(pp=2, cache=True)
        planner = ParallelExecutionPlanner(runtime.cluster, runtime_plan=runtime.plan)
        self.dag(runtime, layer=1, planner=planner)
        self.dag(runtime, layer=1, iteration=1, planner=planner)
        cached = self.dag(runtime, layer=1, iteration=2, planner=planner)
        reference = self.dag(self.runtime(pp=2), layer=1, iteration=2)
        self.assertEqual(cached.nodes, reference.nodes)
        self.assertEqual(planner.template_cache.hits, 1)
        service = self.execute(runtime)
        self.assertTrue(
            any(
                event.operation == "af_pipeline_handoff"
                for event in service.trace.events
            )
        )
        settings = dict(
            batch=1,
            queries=1,
            phase="decode",
            emit_logits=True,
            context=17,
            dp_rank=0,
            seed=42,
            sample=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = cached_profile(runtime, "reuse", Path(directory), **settings)
            again = cached_profile(runtime, "reuse", Path(directory), **settings)
            self.assertEqual(profile.nodes, again.nodes)
            profile.verify()
            baseline = cached_profile(
                self.runtime(False, pp=2), "reuse", Path(directory), **settings
            )
            self.assertNotEqual(profile.nodes, baseline.nodes)

    def test_overlapping_iterations_keep_causal_dependencies_and_resource_capacity(
        self,
    ):
        runtime = self.runtime(cache=True)
        env = simpy.Environment()
        service = AFService(env, runtime, seed=42, retain_trace=True)
        for i in range(2):
            session = service.admit(i, 0, maximum_context=24, current_context=16)
            session.generated = 1
            env.process(service.forward([session], phase="decode"))
        env.run()
        for iteration in (0, 1):
            service.trace.validate_invariants(iteration)
        self.assertFalse(
            any(event.operation == "af_a_to_f" for event in service.trace.events)
        )

    def test_pure_moe_layers_preserve_every_native_trace_event(self):
        model = self.hardware.model("gpt-oss-2t-expert-proxy")
        model = replace(
            model,
            hidden_size=32,
            intermediate_size=64,
            num_layers=1,
            attention_windows=(),
            attention_geometries=(),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            moe=replace(
                model.moe,
                num_experts=8,
                num_moe_layers=1,
                first_k_dense_replace=0,
                moe_intermediate_size=64,
            ),
        )
        reference = self.execute(self.runtime(False, model=model))
        local = self.execute(self.runtime(True, model=model))
        self.assertEqual(reference.trace.events, local.trace.events)

    def test_configuration_requires_colocation_and_strict_boolean(self):
        for value in (1, "true", None):
            with (
                self.subTest(value=value),
                self.assertRaises((ConfigurationError, ValueError)),
            ):
                AFLinkConfig(
                    "a", "a", "f", "af", colocated=True, colocated_dense_reuse=value
                )
        with self.assertRaisesRegex(ConfigurationError, "requires colocated"):
            AFLinkConfig("a", "a", "f", "af", colocated_dense_reuse=True)
