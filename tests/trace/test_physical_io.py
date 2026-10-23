"""Input/output work appears once per owning stage and in physical order."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from TokenSim.af.execution.model import ModelAFExecutor
from tests.trace import test_serving


class PhysicalIOTest(unittest.TestCase):
    def test_tp_input_reduction_and_logits_gather_are_distinct_physical_events(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, result = test_serving.ServingTraceTest().run_cli(
                directory,
                "full",
                "--model",
                "data/models/moe-toy.yaml",
                "--cluster",
                "data/clusters/8_a100/moe_tp2_dp4.json",
            )
            events = list(engine.trace_session.events())
            first = events[0]
            scope = [
                event
                for event in events
                if event.iteration_id == first.iteration_id
                and event.worker_id == first.worker_id
            ]
            operations = [event.operation for event in scope]
            self.assertEqual(operations[:2], ["embedding_lookup", "all_reduce"])
            self.assertEqual(
                operations[-3:], ["lm_head", "all_gather", "token_selection"]
            )
            self.assertEqual(operations.count("embedding_lookup"), 1)
            self.assertEqual(operations.count("all_gather"), 1)
            reductions = [event for event in scope if event.operation == "all_reduce"]
            model = engine.workers[first.worker_id].latency_backend.model
            dense_layers = model.num_layers - len(model.moe_layer_indices())
            self.assertEqual(len(reductions), model.num_layers + dense_layers + 1)
            self.assertIn("dispatch", operations)
            self.assertIn("combine", operations)
            self.assertEqual(reductions[0].details["collective_scope"], "tp_embedding")
            for left, right in zip(scope, scope[1:]):
                self.assertAlmostEqual(left.end_at, right.start_at)

    def test_prefill_chunk_without_logits_has_input_and_norm_but_no_output_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, _ = test_serving.ServingTraceTest().run_cli(
                directory,
                "full",
                "--request_count",
                "1",
                "--batching",
                "paged-attn",
                "--max_num_batched_tokens",
                "8",
                "--prefill_mean_len",
                "32",
                "--prefill_range_len",
                "0",
            )
            events = list(engine.trace_session.recorder.events_for(0))
            names = {event.operation for event in events}
            self.assertIn("embedding_lookup", names)
            self.assertIn("final_norm", names)
            self.assertFalse(names & {"lm_head", "all_gather", "token_selection"})
            for name in ("embedding_lookup", "final_norm"):
                event = next(event for event in events if event.operation == name)
                self.assertEqual(event.details["shape"]["num_tokens"], 8)

    def test_af_output_normalization_covers_forward_rows_without_logits(self):
        executor = object.__new__(ModelAFExecutor)
        executor.total_layers = 2
        executor.hidden_size = 256
        binding = SimpleNamespace(
            domain="a",
            domain_rank=0,
            physical_identity="worker:0",
            dp_rank=0,
            tp_rank=0,
            ep_rank=0,
            tp_group=(0,),
            ep_group=(0,),
        )
        executor.runtime_plan = SimpleNamespace(
            bindings_for_layer=lambda *args: (binding,)
        )
        block = SimpleNamespace(
            placement=SimpleNamespace(a_domain="a", a_instance=0),
            token_count=32,
            logits_rows=0,
            iteration_id=0,
            block_id=0,
        )
        self.assertEqual(
            executor._boundary_nodes(block, "final_norm")[0].local_work.token_count, 32
        )
        self.assertEqual(executor._boundary_nodes(block, "lm_head"), ())
        self.assertEqual(executor._boundary_nodes(block, "token_selection"), ())
