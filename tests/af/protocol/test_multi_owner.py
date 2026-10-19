"""Multiple A owners share native F routing, expert work and combined returns."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.execution.test_batching import submit_owner
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    af_test_hardware,
    af_test_resources,
    FixedComputeProvider,
    FixedTransferProvider,
)
from tests.af.support.model import _moe


class NativeMultiOwnerTest(unittest.TestCase):
    def runtime(self, *, native_control=False, batching=True, max_tokens=0, layers=1):
        cluster = _cluster(a_tp=1, a_dp=2, f_tp=1, f_dp=1, f_ep=1)
        cluster = replace(
            cluster,
            af=replace(
                cluster.af,
                af_block_size=2,
                ffn_batching=batching,
                ffn_batch_max_tokens=max_tokens,
                execution_protocol="vllm_afd",
                communication_timing=replace(
                    cluster.af.communication_timing,
                    enabled=True,
                    native_control_enabled=native_control,
                ),
            ),
        )
        hardware = af_test_hardware(
            layers=layers, capacity_gib=0.1, moe=replace(_moe(), num_moe_layers=layers)
        )
        model = replace(
            hardware.model("TestModel"),
            model_id="qwen3-tiny",
            dtype="bf16",
            kv_cache_dtype="bf16",
            qk_norm=True,
        )
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
            dtype="bf16",
        )
        env = simpy.Environment()
        provider = FixedComputeProvider(1e-6, 20e-6, router=1e-6, expert=20e-6)
        transfer = FixedTransferProvider(0, 0, timing_scope="transport_service")
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            provider,
            transfer,
            model.model_id,
            "bf16",
            layers,
            16,
            runtime_plan=plan,
            moe_config=model.moe,
            resources=af_test_resources(env, cluster),
        )
        return env, executor, provider, transfer

    def test_gate_experts_and_combine_batch_across_owners(self):
        env, executor, provider, transfer = self.runtime()
        for owner in range(2):
            requests = [
                SimpleNamespace(
                    id=owner * 100 + i,
                    is_prefill=False,
                    prefill_len=8,
                    context_len=8,
                    generation_idx=0,
                )
                for i in range(2)
            ]
            env.process(executor.run_iteration(executor.submit(owner, requests)))
        env.run()
        for stage, tokens in (
            ("router", 4),
            ("routed_expert", 8),
            ("weighted_combine", 4),
        ):
            queries = [q for q in provider.queries if q.stage == stage]
            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0].local_token_count, tokens)
        returned = [q for q in transfer.queries if q.operation == "af_f_to_a"]
        self.assertEqual(len(returned), 2)
        self.assertEqual([q.payload_bytes for q in returned], [64, 64])
        for iteration_id in (0, 1):
            executor.trace.validate_invariants(iteration_id)

    def test_native_control_remains_restricted_to_calibrated_rank_count(self):
        with self.assertRaisesRegex(ConfigurationError, "native control requires 1A1F"):
            self.runtime(native_control=True)

    def test_late_peer_blocks_f_compute_even_when_optional_batching_is_disabled(self):
        for batching in (False, True):
            with self.subTest(batching=batching):
                env, executor, provider, _ = self.runtime(batching=batching)
                left, _ = submit_owner(executor, 0)
                task = env.process(executor.run_iteration(left))
                env.run(until=100e-6)
                self.assertFalse(task.triggered)
                self.assertFalse(any(q.stage == "router" for q in provider.queries))
                right, _ = submit_owner(executor, 1, count=1)
                env.process(executor.run_iteration(right))
                env.run()
                self.assertTrue(task.triggered)
                router = [q for q in provider.queries if q.stage == "router"]
                self.assertEqual(len(router), 1)
                self.assertEqual(router[0].original_token_count, 3)
                for event in executor.trace.events:
                    if event.stage == "router":
                        self.assertGreaterEqual(event.start_at, 100e-6)
                self.assertFalse(executor.native_batches.rounds)

    def test_rounds_and_microbatches_do_not_mix_or_overtake(self):
        env, executor, provider, _ = self.runtime()
        iterations = {}
        for owner in range(2):
            for round_id in range(2):
                iteration, _ = submit_owner(
                    executor, owner, count=4, offset=round_id * 1000
                )
                iterations[owner, round_id] = iteration

        def run(owner, round_id, delay):
            yield env.timeout(delay)
            yield from executor.run_iteration(iterations[owner, round_id])

        for owner, round_id in iterations:
            env.process(
                run(owner, round_id, 100e-6 if (owner, round_id) == (1, 0) else 0)
            )
        env.run(until=50e-6)
        self.assertFalse(any(q.stage == "router" for q in provider.queries))
        env.run()
        queries = [q for q in provider.queries if q.stage == "router"]
        self.assertEqual(len(queries), 4)
        self.assertEqual(
            [{edge.request_id for edge in query.routing_edges} for query in queries],
            [
                {0, 1, 100, 101},
                {2, 3, 102, 103},
                {1000, 1001, 1100, 1101},
                {1002, 1003, 1102, 1103},
            ],
        )
        self.assertTrue(all(it.end_at is not None for it in iterations.values()))
        self.assertFalse(executor.native_batches.iterations)

    def test_inconsistent_peer_microbatches_abort_waiting_owner(self):
        env, executor, _, _ = self.runtime()
        errors = []

        def guarded():
            try:
                yield from executor.run_iteration(submit_owner(executor, 0)[0])
            except SimulationStateError as error:
                errors.append(str(error))

        env.process(guarded())
        env.run(until=100e-6)
        with self.assertRaisesRegex(ConfigurationError, "matching microbatch counts"):
            submit_owner(executor, 1, count=4)
        env.run()
        self.assertEqual(len(errors), 1)
        self.assertFalse(executor.active_iterations)
        self.assertFalse(executor.ffn_batcher.tasks)

    def test_cancelling_one_native_peer_aborts_the_transaction(self):
        for cancel_at in (2e-6, 10e-6):
            with self.subTest(cancel_at=cancel_at):
                env, executor, _, _ = self.runtime()
                failures = []

                def guarded(iteration):
                    try:
                        yield from executor.run_iteration(iteration)
                    except (SimulationStateError, simpy.Interrupt) as error:
                        failures.append(error)

                tasks = [
                    env.process(guarded(submit_owner(executor, owner)[0]))
                    for owner in range(2)
                ]

                def cancel():
                    yield env.timeout(cancel_at)
                    tasks[0].interrupt()

                env.process(cancel())
                env.run()
                self.assertEqual(len(failures), 2)
                self.assertFalse(executor.active_iterations)
                self.assertFalse(executor.active_stages)
                self.assertFalse(executor.ffn_batcher.tasks)
                self.assertTrue(
                    all(
                        not r.count and not r.queue
                        for r in executor.resources.compute_slots.values()
                    )
                )
                with self.assertRaisesRegex(SimulationStateError, "new executor"):
                    submit_owner(executor, 0, offset=1000)

    def test_native_batch_limit_cannot_split_a_peer_transaction(self):
        env, executor, provider, _ = self.runtime(max_tokens=2)
        errors = []

        def guarded(iteration):
            try:
                yield from executor.run_iteration(iteration)
            except SimulationStateError as error:
                errors.append(str(error))

        for owner in range(2):
            env.process(guarded(submit_owner(executor, owner)[0]))
        env.run()
        self.assertEqual(len(errors), 2)
        self.assertFalse(any(q.stage == "router" for q in provider.queries))
        self.assertTrue(any("workspace budget" in error for error in errors))

    def test_f_layer_microbatch_order_spans_multiple_layers(self):
        env, executor, provider, _ = self.runtime(layers=2)
        for owner in range(2):
            env.process(
                executor.run_iteration(submit_owner(executor, owner, count=4)[0])
            )
        env.run()
        routers = [q for q in provider.queries if q.stage == "router"]
        self.assertEqual(
            [(q.layer_id, q.block_id) for q in routers],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertFalse(executor.native_batches.rounds)
        self.assertFalse(executor.ffn_batcher.changed)

    def test_peer_failure_before_router_aborts_other_owner(self):
        env, executor, provider, _ = self.runtime()
        original = provider.estimate

        def fail(query):
            if query.stage == "attention" and query.dp_rank == 1:
                raise ValueError("peer attention failed")
            return original(query)

        provider.estimate = fail
        errors = []

        def guarded(iteration):
            try:
                yield from executor.run_iteration(iteration)
            except SimulationStateError as error:
                errors.append(str(error))

        for owner in range(2):
            env.process(guarded(submit_owner(executor, owner)[0]))
        env.run()
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("peer attention failed" in error for error in errors))
        self.assertFalse(executor.active_iterations)
        self.assertFalse(executor.ffn_batcher.changed)

    def test_fixed_batch_helper_rejects_missing_native_peers(self):
        from TokenSim.af.factory import simulate_decode

        _, executor, _, _ = self.runtime()
        runtime = SimpleNamespace(cluster=executor.cluster_config)
        with self.assertRaisesRegex(ConfigurationError, "submit every peer"):
            simulate_decode(runtime, 2, 8)
