from dataclasses import replace
import unittest
from unittest.mock import patch

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.factory import build_model_runtime
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.errors import ConfigurationError, OutOfBlocksError, SimulationStateError
from TokenSim.operator_data.workload import work_for_table
from scripts.deepseek_v3_runtime import configuration, simulate
from tests.af.support.runtime import _request


def runtime(full=False, a_tp=1, f_tp=2, block=8, hypothetical=None):
    config = configuration(
        full_model=full,
        a_tp=a_tp,
        f_tp=f_tp,
        af_batch_size=block,
        hypothetical_f_memory_gib=hypothetical,
    )
    config.doc["af"]["router_location"] = "f"
    config.doc["af"]["communication_timing"] = dict(
        enabled=True, host_submit_us=5, host_completion_us=0
    )
    return build_model_runtime(config.doc, hardware=config.hardware, model=config.model)


def run(config, batch, context, steps=1):
    iterations = []
    original = ModelAFExecutor.submit

    def submit(executor, *args, **kwargs):
        iteration = original(executor, *args, **kwargs)
        iterations.append(iteration)
        return iteration

    with patch.object(ModelAFExecutor, "submit", submit):
        executor, _, _, _ = simulate(
            config, batch, context, steps=steps, fallback="analytical_only"
        )
    return executor, iterations


class FullDecodeTest(unittest.TestCase):
    def test_multi_a_routes_weights_to_every_combine_owner_and_runs_parallel(self):
        for a_tp in (2, 4, 8):
            with self.subTest(a_tp=a_tp):
                config = runtime(a_tp=a_tp, f_tp=2, block=2)
                executor, iterations = run(config, 4, 32, 2)
                for iteration in iterations:
                    for block in iteration.blocks:
                        events = [
                            e
                            for e in executor.trace.events
                            if e.iteration_id == iteration.iteration_id
                            and e.block_id == block.block_id
                            and e.layer_id == 0
                        ]
                        attention = [e for e in events if e.stage == "attention"]
                        self.assertEqual(len(attention), a_tp)
                        self.assertEqual(len({e.start_at for e in attention}), 1)
                        self.assertGreater(
                            min(e.end_at for e in attention),
                            max(e.start_at for e in attention),
                        )
                        metadata = {
                            e.target_endpoint.domain_rank: e
                            for e in events
                            if e.operation == "af_router_metadata"
                            and e.target_domain == "a-domain"
                        }
                        self.assertEqual(set(metadata), set(range(a_tp)))
                        for combine in (
                            e for e in events if e.stage == "weighted_combine"
                        ):
                            arrival = metadata[combine.participant_ranks[0]]
                            self.assertGreaterEqual(combine.start_at, arrival.end_at)
                            self.assertEqual(
                                arrival.payload_bytes, block.token_count * 8 * 8
                            )
                        boundary = [
                            e
                            for e in executor.trace.events
                            if e.iteration_id == iteration.iteration_id
                            and e.block_id == block.block_id
                            and e.layer_id < 0
                            and e.event_type == "compute"
                        ]
                        self.assertEqual(len(boundary), 4 * a_tp)
                        lm_heads = [
                            c
                            for e in boundary
                            if e.operation == "lm_head"
                            for c in e.details["components"]
                        ]
                        self.assertTrue(lm_heads)
                        self.assertTrue(
                            all(c["key"]["n"] == 129280 // a_tp for c in lm_heads)
                        )
                        comm = [
                            e
                            for e in executor.trace.events
                            if e.iteration_id == iteration.iteration_id
                            and e.block_id == block.block_id
                            and e.layer_id < 0
                            and e.event_type == "collective"
                        ]
                        self.assertEqual(
                            {e.operation for e in comm}, {"all_reduce", "all_gather"}
                        )
                        gather = next(e for e in comm if e.operation == "all_gather")
                        self.assertEqual(
                            gather.payload_bytes,
                            block.token_count * (129280 // a_tp) * 2,
                        )
                        self.assertEqual(
                            set(gather.participant_ranks), set(range(a_tp))
                        )
                for iteration in iterations:
                    executor.trace.validate_invariants(iteration.iteration_id)
                # Corrupt a dependency to prove boundary ordering is checked by core trace validation.
                event = next(
                    e for e in executor.trace.events if e.operation == "lm_head"
                )
                index = executor.trace.events.index(event)
                executor.trace.events[index] = replace(event, start_at=-1)
                with self.assertRaises(SimulationStateError):
                    executor.trace.validate_invariants(event.iteration_id)

    def test_full_multi_a_capacity_retains_replicated_mla_cache(self):
        one = runtime(True, 1, 32).plan
        multi = runtime(True, 4, 32).plan
        a_states = [
            s
            for s in multi.physical_states
            if s.physical_identity.startswith("a-pool:")
        ]
        baseline = next(
            s for s in one.physical_states if s.physical_identity == "a-pool:0"
        )
        self.assertEqual(len(a_states), 4)
        for state in a_states:
            self.assertEqual(state.kv_bytes_per_block, baseline.kv_bytes_per_block)
            self.assertGreater(state.model_bytes, baseline.model_bytes / 4)
            self.assertLess(state.model_bytes, baseline.model_bytes)
        self.assertTrue(
            all(s.model_bytes < s.capacity_bytes for s in multi.physical_states)
        )

    def test_routing_does_not_change_when_af_blocks_change(self):
        results = []
        for block in (8, 32):
            executor, _ = run(runtime(block=block), 32, 32)
            # Preserve the old request-to-expert invariant across AF block sizes.
            # Expert choice depends on token identity, not block position.
            edges = {
                edge for event in executor.trace.events for edge in event.routing_edges
            }
            results.append(edges)
        self.assertTrue(results[0])
        self.assertEqual(results[0], results[1])

    def test_f_leader_metadata_and_model_boundaries_contend_and_order(self):
        config = runtime(block=32)
        executor, iterations = run(config, 64, 32, 2)
        boundary = [
            e
            for e in executor.trace.events
            if e.layer_id < 0 and e.event_type == "compute"
        ]
        self.assertEqual(len(boundary), 16)
        self.assertGreaterEqual(iterations[1].start_at, iterations[0].end_at)
        for iteration in iterations:
            for block in iteration.blocks:
                events = [
                    e
                    for e in executor.trace.events
                    if e.iteration_id == iteration.iteration_id
                    and e.block_id == block.block_id
                ]
                router = next(e for e in events if e.stage == "router")
                self.assertEqual(router.physical_endpoints, ("f-pool:0",))
                metadata = next(
                    e for e in events if e.operation == "af_router_metadata"
                )
                self.assertEqual(metadata.payload_bytes, block.token_count * 8 * 8)
                self.assertEqual(metadata.direction, "f_to_f")
                self.assertEqual(metadata.source_endpoint.physical_worker, "f-pool:0")
                self.assertEqual(metadata.target_endpoint.physical_worker, "f-pool:1")
                self.assertGreaterEqual(metadata.start_at, router.end_at)
                self.assertIsNone(metadata.dtype)  # mixed int32/FP32 protocol
                arrivals = [e for e in events if e.operation == "af_a_to_f"]
                self.assertEqual(len(arrivals), 2)
                self.assertTrue(
                    all(
                        e.payload_bytes == block.token_count * 7168 * 2
                        for e in arrivals
                    )
                )
                f_input = next(
                    e
                    for e in arrivals
                    if e.target_endpoint.physical_worker == "f-pool:0"
                )
                self.assertGreaterEqual(router.start_at, f_input.end_at)
                remote_experts = [
                    e
                    for e in events
                    if e.stage == "routed_expert"
                    and e.physical_endpoints == ("f-pool:1",)
                ]
                self.assertTrue(remote_experts)
                self.assertTrue(
                    all(e.start_at >= metadata.end_at for e in remote_experts)
                )
                a_metadata = next(
                    e
                    for e in events
                    if e.operation == "af_router_metadata"
                    and e.target_domain == "a-domain"
                )
                combine = next(e for e in events if e.stage == "weighted_combine")
                self.assertEqual(a_metadata.payload_bytes, block.token_count * 8 * 8)
                self.assertGreaterEqual(combine.start_at, a_metadata.end_at)
        for iteration in iterations:
            executor.trace.validate_invariants(iteration.iteration_id)

    def test_actual_full_model_capacity_and_layer_types(self):
        config = runtime(True, 1, 32)
        plan = config.plan
        self.assertEqual(
            [x.layer_type for x in plan.layers], ["dense"] * 3 + ["moe"] * 58
        )
        self.assertTrue(
            all(p.model_bytes < p.capacity_bytes for p in plan.physical_states)
        )
        # The serving ledger exposes per-rank KV limits, including replicated latent cache.
        replica = next(r for r in plan.replicas if r.domain == "a-domain")
        self.assertLess(replica.logical_gpu_blocks, 128 * (16384 // plan.block_size))
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        oversized = _request(
            0, prefill_len=(replica.logical_gpu_blocks + 1) * plan.block_size
        )
        with self.assertRaises(OutOfBlocksError):
            manager.allocate(oversized)
        with self.assertRaises(ConfigurationError):
            runtime(True, 1, 1)

    def test_embedding_and_greedy_selection_traffic(self):
        embedding = work_for_table(
            "elementwise",
            dict(
                op_name="embedding_lookup", num_tokens=4, hidden_size=7168, dtype="bf16"
            ),
        )
        argmax = work_for_table(
            "elementwise",
            dict(op_name="argmax", num_tokens=4, hidden_size=129280, dtype="bf16"),
        )
        self.assertEqual(embedding.activation_bytes, 4 * 7168 * 4)
        self.assertEqual(argmax.activation_bytes, 4 * 129280 * 2)
        self.assertEqual(argmax.flops, 4 * 129280)

    def test_hypothetical_f_memory_preserves_a_capacity_and_device_throughput(self):
        config = runtime(True, 1, 1, hypothetical=1536)
        states = {s.physical_identity: s for s in config.plan.physical_states}
        base = config.hardware.device("a100_sxm_80g")
        self.assertEqual(
            states["a-pool:0"].capacity_bytes,
            runtime(True, 1, 32).plan.physical_states[0].capacity_bytes,
        )
        self.assertLess(states["f-pool:0"].capacity_bytes, 1536 * 2**30)
        self.assertEqual(
            config.hardware.device(
                states["f-pool:0"].hardware
            ).memory_capacity_bytes.value,
            1536 * 2**30,
        )
        self.assertGreater(
            states["f-pool:0"].model_bytes, base.memory_capacity_bytes.value
        )
        virtual = config.hardware.device(states["f-pool:0"].hardware)
        self.assertEqual(virtual.peak_compute, base.peak_compute)
        self.assertEqual(
            virtual.memory_bandwidth_bytes_per_s, base.memory_bandwidth_bytes_per_s
        )
        with self.assertRaises(ConfigurationError):
            runtime(True, 1, 1, hypothetical=80)


if __name__ == "__main__":
    unittest.main()
