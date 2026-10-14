"""Ready-input operation service boundary and actual charging oracles."""

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import simpy
from tests import test_af_host_profile as host
from tests import test_af_fixed_host as fixed
from TokenSim.af.calibration import runtime_fingerprint
from TokenSim.af.operation_profile import OperationServiceProfile
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.errors import ConfigurationError


class OperationProfileTest(unittest.TestCase):
    def test_v3_reserves_batch5(self):
        profile = self.document(self.host.runtime())
        profile["schema_version"] = 3
        profile["evidence"]["heldout_batches"] = [5]
        self.load(profile)
        for version, heldout in ((3, [2]), (2, [5]), (1, [5])):
            profile["schema_version"] = version
            profile["evidence"]["heldout_batches"] = heldout
            with self.assertRaises(ConfigurationError):
                self.load(profile)

    @classmethod
    def setUpClass(cls):
        host.HostProfileTest.setUpClass()
        fixed.FixedHostTest.setUpClass()
        cls.host = host.HostProfileTest()
        cls.fixed = fixed.FixedHostTest()

    def document(self, runtime):
        doc = json.loads(Path("tests/fixtures/af_operation_service.json").read_text())
        doc["runtime_fingerprint"] = runtime_fingerprint(
            runtime.plan, runtime.cluster, runtime.workers, runtime.hardware
        )
        doc["environment"] = {
            "execution_mode": "eager",
            "software_sha256": "test-software",
        }
        doc["evidence"]["kind"] = "synthetic_test"
        for op in doc["operations"].values():
            op["fixed_us"] = 11
        return doc

    def load(self, doc, explicit=()):
        runtime = self.host.runtime()
        return OperationServiceProfile(
            doc,
            expected_runtime=runtime_fingerprint(
                runtime.plan, runtime.cluster, runtime.workers, runtime.hardware
            ),
            expected_environment={
                "execution_mode": "eager",
                "software_sha256": "test-software",
            },
            explicit_costs=explicit,
        )

    def configured(self, profile=None, **options):
        runtime = self.host.runtime()
        doc = copy.deepcopy(runtime.doc)
        profile = profile or self.document(runtime)
        doc["af"].update(
            operation_service_profile=profile,
            calibration_environment=profile["environment"],
            **options,
        )
        return build_model_runtime(doc, hardware=runtime.hardware, model=runtime.model)

    def test_rejects_invalid_scope_and_evidence(self):
        original = self.document(self.host.runtime())
        edits = [
            lambda d: d.update(timing_scope="host_service"),
            lambda d: d.update(runtime_fingerprint="wrong"),
            lambda d: d.update(environment={}),
            lambda d: d["operations"]["h2d"].update(coverage=5),
            lambda d: d["operations"]["h2d"].update(coverage=["host_submit"]),
            lambda d: d["operations"]["h2d"].update(fixed_us=float("nan")),
            lambda d: d["operations"]["h2d"].update(fixed_us=-1),
            lambda d: d["operations"]["h2d"].update(fixed_us=True),
            lambda d: d["operations"]["h2d"].update(completion="api_enqueue"),
            lambda d: d["evidence"].update(external_wait_excluded=False),
            lambda d: d["evidence"].update(training_batches=1),
            lambda d: d["evidence"].update(heldout_batches=None),
            lambda d: d["evidence"].update(windows=2),
        ]
        for edit in edits:
            doc = copy.deepcopy(original)
            edit(doc)
            with self.subTest(doc=doc), self.assertRaises(ConfigurationError):
                self.load(doc)
        for cost in (
            "control_h2d_us",
            "control_length_readback_us",
            "control_payload_readback_us",
        ):
            with self.assertRaisesRegex(ConfigurationError, "duplicate"):
                self.load(original, (cost,))

    def sampled_profile(self):
        doc = self.document(self.host.runtime())
        doc["schema_version"] = 2
        doc["evidence"]["heldout_batches"] = [2]
        doc["operations"]["sampled_token_readback"] = dict(
            fixed_us=11,
            coverage=["sampled_token_readback"],
            input_ready="sampled_tokens_device_ready",
            completion="cpu_api_return_data_ready",
            resource_model="calling_thread_single_inflight",
        )
        doc["evidence"]["statistics"]["sampled_token_readback"] = dict(
            p50_us=11, p95_us=11, samples=100
        )
        return doc

    def test_sampled_readback_gates_each_step_completion_and_preserves_old_run(self):
        baseline = self.document(self.host.runtime())
        _, before, _, _ = simulate_decode(
            self.configured(baseline), 1, 129, steps=3, fallback="analytical_only"
        )
        ex, after, _, _ = simulate_decode(
            self.configured(self.sampled_profile()),
            1,
            129,
            steps=3,
            fallback="analytical_only",
        )
        for old, new in zip(before, after):
            self.assertAlmostEqual(new["makespan_us"] - old["makespan_us"], 11)
        outputs = [
            e for e in ex.trace.events if e.operation == "sampled_token_readback"
        ]
        self.assertEqual(len(outputs), 3)
        for event in outputs:
            sample = next(
                e
                for e in ex.trace.events
                if e.iteration_id == event.iteration_id
                and e.event_type == "compute"
                and e.operation == "token_selection"
            )
            self.assertGreaterEqual(event.start_at, sample.end_at)
            self.assertIn(sample.node_id, event.predecessor_ids)
            self.assertEqual(event.details["message_class"], "engine_output")
            self.assertAlmostEqual(event.service_duration * 1e6, 11)
            self.assertAlmostEqual(
                event.end_at,
                max(
                    e.end_at
                    for e in ex.trace.events
                    if e.iteration_id == event.iteration_id
                ),
            )
            ex.trace.validate_invariants(event.iteration_id)

    def test_readback_v2_rejects_legacy_version_or_wrong_ready(self):
        doc = self.sampled_profile()
        doc["schema_version"] = 1
        with self.assertRaises(ConfigurationError):
            self.load(doc)
        doc = self.sampled_profile()
        doc["operations"]["sampled_token_readback"]["input_ready"] = "api_enqueue"
        with self.assertRaises(ConfigurationError):
            self.load(doc)

    def test_readback_cache_and_trace_switches_preserve_completion(self):
        rows = []
        for enabled in (True, False):
            runtime = self.configured(
                self.sampled_profile(),
                trace_enabled=enabled,
                service_estimate_cache=enabled,
                dag_template_cache=enabled,
            )
            _, result, _, _ = simulate_decode(
                runtime, 1, 129, steps=2, fallback="analytical_only", trace=enabled
            )
            rows.append(result)
        self.assertEqual(rows[0], rows[1])

    def test_sampled_ready_100_or_120_plus_11(self):
        for ready in (100, 120):
            env, ex, block, node = self.oracle()
            ex.operation_profile = self.load(self.sampled_profile())

            def run():
                yield env.timeout(ready / 1e6)
                yield from ex._control_stage(
                    block,
                    node,
                    "sampled_token_readback",
                    (),
                    node.physical_workers[0],
                    0,
                    device=True,
                )

            env.process(run())
            env.run()
            self.assertAlmostEqual(env.now * 1e6, ready + 11)
            event = next(
                e for e in ex.trace.events if e.operation == "sampled_token_readback"
            )
            self.assertAlmostEqual(event.service_duration * 1e6, 11)

    def test_workload_and_wire_size_checked(self):
        doc = self.document(self.host.runtime())
        for count, context, phase in (
            (9, 129, "decode"),
            (1, 128, "decode"),
            (1, 129, "prefill"),
        ):
            with self.assertRaises(ConfigurationError):
                self.load(doc).validate_workload(
                    [SimpleNamespace(context_len=context)] * count, phase
                )
        doc["payload_bytes"] += 1
        with self.assertRaisesRegex(ConfigurationError, "wire size"):
            self.load(doc).validate_workload(
                [SimpleNamespace(context_len=129)], "decode"
            )

    def test_explicit_zero_conflicts_and_legacy_device_events_remain(self):
        runtime = self.host.runtime({"control_h2d_us": 0})
        doc = copy.deepcopy(runtime.doc)
        profile = self.document(runtime)
        doc["af"].update(
            operation_service_profile=profile,
            calibration_environment=profile["environment"],
        )
        configured = build_model_runtime(
            doc, hardware=runtime.hardware, model=runtime.model
        )
        with self.assertRaisesRegex(ConfigurationError, "duplicate"):
            simulate_decode(configured, 1, 129, fallback="analytical_only")
        ex, _, _, _ = simulate_decode(runtime, 1, 129, fallback="analytical_only")
        self.assertEqual(
            len([e for e in ex.trace.events if e.event_type == "control_device"]), 3
        )
        self.assertFalse(
            any(e.event_type == "operation_service" for e in ex.trace.events)
        )

    def test_standard_engine_loads_operation_profile(self):
        from unittest.mock import patch
        from tests.test_af_executor import _psla_config
        from TokenSim.config.config import ClusterConfig, KVTransferConfig
        from TokenSim.llm.llm_engine import LLMEngine

        runtime = self.configured(self.sampled_profile())
        env = simpy.Environment()
        with patch.object(runtime.hardware, "model", return_value=runtime.model):
            engine = LLMEngine(
                env=env,
                block_size=16,
                batching="paged-attn",
                kv_transfer_config=KVTransferConfig.default(),
                psla_config=_psla_config(),
                cluster_config=ClusterConfig(**runtime.doc),
                hardware=runtime.hardware,
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_parallem_sum=8,
                latency_fallback="analytical_only",
            )
        ex = engine.af_executor
        self.fixed.run_step(env, ex)
        self.assertEqual(
            len([e for e in ex.trace.events if e.event_type == "operation_service"]), 4
        )
        self.assertEqual(
            len(
                [e for e in ex.trace.events if e.operation == "sampled_token_readback"]
            ),
            1,
        )

    def test_cancel_sampled_readback_releases_host_and_can_resume(self):
        profile = self.sampled_profile()
        profile["operations"]["sampled_token_readback"]["fixed_us"] = 1000000
        ex, rows, _, _ = simulate_decode(
            self.configured(profile), 1, 129, fallback="analytical_only"
        )
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = ex.env.process(ex.run_iteration(ex.submit(0, [request], "decode")))
        ex.env.run(until=ex.env.now + (rows[0]["makespan_us"] - 1000000 + 5) / 1e6)
        self.assertTrue(
            any(r.users for r in ex.resources.host_submission_slots.values())
        )
        process.interrupt("cancel token readback")
        with self.assertRaises(simpy.Interrupt):
            ex.env.run(until=process)
        ex.env.run()
        self.assertTrue(
            all(
                not r.users and not r.queue
                for r in ex.resources.host_submission_slots.values()
            )
        )
        self.fixed.run_step(ex.env, ex)

    def test_charges_once_replaces_device_and_keeps_codec(self):
        runtime = self.host.runtime()
        codec = self.host.document(runtime)
        ex, _, _, _ = simulate_decode(
            self.configured(engine_overhead_profile=codec),
            1,
            129,
            steps=3,
            fallback="analytical_only",
        )
        operations = [e for e in ex.trace.events if e.event_type == "operation_service"]
        self.assertEqual(len(operations), 9)
        self.assertFalse(any(e.event_type == "control_device" for e in ex.trace.events))
        self.assertEqual(
            len(
                [
                    e
                    for e in ex.trace.events
                    if e.details.get("cost_field") == "control_encode_us"
                ]
            ),
            3,
        )
        self.assertEqual(
            len(
                [
                    e
                    for e in ex.trace.events
                    if e.details.get("cost_field") == "control_decode_us"
                ]
            ),
            3,
        )
        for event in operations:
            self.assertAlmostEqual(event.service_duration, 11e-6)
            self.assertTrue(event.resource_id.startswith("host:"))
            self.assertEqual(event.details["timing_scope"], "operation_service")
        for i in {e.iteration_id for e in operations}:
            ex.trace.validate_invariants(i)

    def test_cache_identity_and_trace_equivalence(self):
        profile = self.document(self.host.runtime())
        results = []
        namespaces = []
        for enabled in (True, False):
            runtime = self.configured(
                profile,
                trace_enabled=enabled,
                service_estimate_cache=enabled,
                dag_template_cache=enabled,
            )
            _, rows, compute, _ = simulate_decode(
                runtime, 1, 129, fallback="analytical_only", trace=enabled
            )
            results.append(rows)
            namespaces.append(compute._estimate_namespace)
        self.assertEqual(results[0], results[1])
        profile["operations"]["h2d"]["fixed_us"] = 12
        _, changed, compute, _ = simulate_decode(
            self.configured(profile), 1, 129, fallback="analytical_only"
        )
        self.assertNotEqual(namespaces[0], compute._estimate_namespace)
        self.assertGreater(changed[0]["makespan_us"], results[0][0]["makespan_us"])

    def oracle(self):
        env, ex = self.fixed.executor(native_control_enabled=True)
        ex.operation_profile = self.load(self.document(self.host.runtime()))
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        block = ex.submit(0, [request], "decode").blocks[0]
        node = ex._control_nodes(block)[0]
        return env, ex, block, node

    def test_ready_100_plus_11_and_external_delay(self):
        for ready in (100, 180):
            env, ex, block, node = self.oracle()

            def run():
                yield env.timeout(ready / 1e6)
                yield from ex._control_stage(
                    block,
                    node,
                    "length_readback",
                    (),
                    node.physical_workers[1],
                    0,
                    device=True,
                )

            env.run(until=env.process(run()))
            event = ex.trace.events[-1]
            self.assertAlmostEqual(event.end_at * 1e6, ready + 11)
            self.assertAlmostEqual(event.service_duration * 1e6, 11)

    def test_calling_thread_serializes_without_locking_gpu_or_fabric(self):
        env, ex, block, node = self.oracle()
        identity = node.physical_workers[1]
        for name in ("length_readback", "payload_readback"):
            env.process(
                ex._control_stage(block, node, name, (), identity, 0, device=True)
            )
        env.run(until=1e-6)
        slots = ex.resources.host_resources_for((identity,))
        self.assertEqual(sum(len(r.users) for _, r in slots), 1)
        self.assertEqual(sum(len(r.queue) for _, r in slots), 1)
        self.assertTrue(all(not r.users for r in ex.resources.compute_slots.values()))
        self.assertTrue(
            all(not r.users for r in ex.resources.communication_resources.values())
        )
        env.run()
        self.assertAlmostEqual(env.now, 22e-6)

    def test_cancel_releases_operation_resource(self):
        env, ex, block, node = self.oracle()
        process = env.process(
            ex._control_stage(
                block, node, "h2d", (), node.physical_workers[0], 0, device=True
            )
        )
        env.run(until=1e-6)
        process.interrupt("cancel")
        with self.assertRaises(simpy.Interrupt):
            env.run(until=process)
        env.run()
        self.assertTrue(
            all(
                not r.users and not r.queue
                for r in ex.resources.host_submission_slots.values()
            )
        )
        self.fixed.run_step(env, ex)

    def input_profile(self):
        profile = self.document(self.host.runtime())
        profile["operations"]["a_input_prepare"] = dict(
            fixed_us=100,
            coverage=["a_input_prepare"],
            input_ready="prior_device_work_complete",
            completion="device_synchronize_after_operation",
            resource_model="calling_thread_single_inflight",
        )
        profile["evidence"]["statistics"]["a_input_prepare"] = dict(
            p50_us=100, p95_us=101, samples=200
        )
        return profile

    def test_input_preparation_once_gates_control_and_compute(self):
        profile = self.input_profile()
        ex, rows, _, _ = simulate_decode(
            self.configured(profile), 1, 129, steps=2, fallback="analytical_only"
        )
        baseline = simulate_decode(
            self.configured(), 1, 129, steps=2, fallback="analytical_only"
        )[1]
        for row, before in zip(rows, baseline):
            self.assertAlmostEqual(row["makespan_us"] - before["makespan_us"], 100)
            events = ex.trace.events_for(row["iteration_id"])
            prepared = [e for e in events if e.operation == "a_input_prepare"]
            self.assertEqual(len(prepared), 1)
            end = prepared[0].end_at
            for event in events:
                if (
                    event.event_type == "compute"
                    or event.details.get("cost_field") == "control_encode_us"
                ):
                    self.assertGreaterEqual(event.start_at, end)
            ex.trace.validate_invariants(row["iteration_id"])

    def test_input_preparation_rejects_host_overlap_and_unknown_provider_coverage(self):
        profile = self.input_profile()
        codec = self.host.document(self.host.runtime())
        with self.assertRaisesRegex(ConfigurationError, "duplicate"):
            simulate_decode(
                self.configured(profile, engine_overhead_profile=codec),
                1,
                129,
                fallback="analytical_only",
            )
        with self.assertRaisesRegex(ConfigurationError, "analytical_only"):
            simulate_decode(self.configured(profile), 1, 129, fallback="table_first")

    def test_cancel_input_preparation_can_resume(self):
        runtime = self.configured(self.input_profile())
        ex, _, _, _ = simulate_decode(runtime, 1, 129, fallback="analytical_only")
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = ex.env.process(ex.run_iteration(ex.submit(0, [request], "decode")))
        ex.env.run(until=ex.env.now + 1e-6)
        process.interrupt("cancel input prepare")
        with self.assertRaises(simpy.Interrupt):
            ex.env.run(until=process)
        ex.env.run()
        self.assertFalse(ex._step_preparation)
        self.assertTrue(
            all(
                not r.users and not r.queue
                for r in ex.resources.host_submission_slots.values()
            )
        )
        self.fixed.run_step(ex.env, ex)

    def test_metadata_preparation_obeys_native_order(self):
        profile = self.input_profile()
        for name, value in (("a_attention_metadata", 17), ("f_dp_metadata", 13)):
            profile["operations"][name] = dict(
                fixed_us=value,
                coverage=[name],
                input_ready=(
                    "prior_device_work_complete"
                    if name.startswith("a_")
                    else "control_decoded_cpu_metadata_ready"
                ),
                completion=(
                    "device_synchronize_after_operation"
                    if name.startswith("a_")
                    else "cpu_api_return_data_ready"
                ),
                resource_model="calling_thread_single_inflight",
            )
            profile["evidence"]["statistics"][name] = dict(
                p50_us=value, p95_us=value, samples=200
            )
        ex, _, _, _ = simulate_decode(
            self.configured(profile), 1, 129, fallback="analytical_only"
        )
        events = ex.trace.events
        by_op = {e.operation: e for e in events if e.event_type == "operation_service"}
        self.assertEqual(len(by_op), 6)
        self.assertGreaterEqual(
            by_op["a_attention_metadata"].start_at, by_op["a_input_prepare"].end_at
        )
        decoded = next(
            e for e in events if e.details.get("cost_field") == "control_decode_us"
        )
        self.assertGreaterEqual(by_op["f_dp_metadata"].start_at, decoded.end_at)
        first_receive = next(
            e
            for e in events
            if e.node_id.endswith(":receive_submit") and e.layer_id == 0
        )
        self.assertGreaterEqual(first_receive.start_at, by_op["f_dp_metadata"].end_at)
        self.assertIn(by_op["f_dp_metadata"].node_id, first_receive.predecessor_ids)
        ex.trace.validate_invariants(0)


if __name__ == "__main__":
    unittest.main()
