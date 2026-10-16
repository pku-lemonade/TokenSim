"""Template admission, bounded probation and model-independent equivalence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.af.caching.templates import BoundDAG, DAGTemplate, TemplateCache
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.errors import ConfigurationError
from tests.af.support.profiles import NativeFixture
from tests.support.hardware import make_model


class TemplateAdmissionTest(NativeFixture, unittest.TestCase):
    def dense_planner(self):
        runtime = build_model_runtime(
            self.base.doc,
            hardware=self.base.hardware,
            model=make_model(num_layers=1),
        )
        executor, _, _, _ = simulate_decode(runtime, 1, 8, fallback="analytical_only")
        iteration = executor.submit(
            0,
            [
                SimpleNamespace(
                    id=7,
                    is_prefill=False,
                    prefill_len=8,
                    context_len=8,
                    generation_idx=1,
                )
            ],
            "decode",
        )
        planner = executor.plan_builder
        planner.template_cache = TemplateCache(limit=2)
        return planner, iteration.blocks[0], runtime.cluster.af.links[0]

    def test_first_shape_skips_compilation_and_reuse_binds_current_identities(self):
        planner, block, link = self.dense_planner()
        with patch.object(DAGTemplate, "compile", wraps=DAGTemplate.compile) as compile:
            first = planner.build_layer(block, 0, link)
            self.assertIsInstance(first, AFExecutionDAG)
            compile.assert_not_called()
            changed = replace(block, iteration_id=block.iteration_id + 1, block_id=3)
            second = planner.build_layer(changed, 0, link)
            self.assertIsInstance(second, BoundDAG)
            self.assertEqual(compile.call_count, 1)
            third_block = replace(changed, iteration_id=changed.iteration_id + 1)
            third = planner.build_layer(third_block, 0, link)
            self.assertEqual(compile.call_count, 1)
        for current, dag in ((block, first), (changed, second), (third_block, third)):
            reference = planner._build_runtime_layer(current, 0, link)
            self.assertEqual(dag.nodes, reference.nodes)
            self.assertEqual(dag.terminal_node_ids, reference.terminal_node_ids)
        self.assertEqual(planner.template_cache.hits, 1)

    def test_unique_shapes_keep_only_bounded_probation_keys(self):
        planner, block, link = self.dense_planner()
        cache = planner.template_cache
        with patch.object(DAGTemplate, "compile", wraps=DAGTemplate.compile) as compile:
            for count in (1, 2, 3, 1):
                changed = replace(
                    block,
                    request_ids=tuple(range(count)),
                    request_count=count,
                    token_count=count,
                    context_lengths=(8,) * count,
                    query_lengths=(1,) * count,
                    query_offsets=(7,) * count,
                )
                self.assertIsInstance(
                    planner.build_layer(changed, 0, link), AFExecutionDAG
                )
                self.assertLessEqual(len(cache.probation), 2)
                self.assertEqual(len(cache.templates), 0)
            compile.assert_not_called()
        self.assertEqual(cache.hits, 0)

    def test_failed_reference_validation_does_not_admit_a_shape(self):
        planner, block, link = self.dense_planner()
        with patch.object(
            planner,
            "_build_runtime_layer",
            side_effect=ConfigurationError("invalid graph"),
        ):
            with self.assertRaisesRegex(ConfigurationError, "invalid graph"):
                planner.build_layer(block, 0, link)
        self.assertEqual(len(planner.template_cache.probation), 0)
        self.assertEqual(len(planner.template_cache.templates), 0)

    def test_dense_mha_and_gqa_moe_preserve_complete_json_results(self):
        for model in (make_model(num_layers=2), self.model):
            with self.subTest(model=model.model_id), TemporaryDirectory() as directory:
                results = []
                for enabled in (False, True):
                    document = deepcopy(self.base.doc)
                    document["af"]["dag_template_cache"] = enabled
                    runtime = build_model_runtime(
                        document, hardware=self.base.hardware, model=model
                    )
                    executor, rows, _, _ = simulate_decode(
                        runtime, 2, 16, steps=4, fallback="analytical_only"
                    )
                    output = Path(directory) / f"{enabled}.json"
                    write_chrome_trace(output, executor.trace.events)
                    results.append((rows, json.loads(output.read_text())))
                    if enabled and not model.moe.is_moe_model:
                        self.assertGreater(executor.plan_builder.template_cache.hits, 0)
                self.assertEqual(results[0], results[1])
