"""Af executor."""

import math
import unittest

import simpy

from TokenSim.config.cluster.core import ClusterConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.support.config import _cluster_dict, _load_cluster, _two_link_cluster
from tests.af.support.execution import _executor, _requests
from tests.af.support.providers import (
    FixedComputeProvider,
    FixedTransferProvider,
    InvalidComputeProvider,
)


class AFExecutorEndToEndTest(unittest.TestCase):
    def test_complete_and_af_compute_share_a_physical_worker_slot(self):
        values = _cluster_dict(phases=("decode",))
        values["worker_groups"][0] = {
            "name": "a-pool",
            "capabilities": ["prefill", "af_attention"],
            "hardware": "a-device",
            "num_workers": 1,
            "network": "af-net",
        }
        values["worker_groups"].pop(1)
        values["num_workers"] = sum(
            group["num_workers"] for group in values["worker_groups"]
        )
        cluster = ClusterConfig(**values)
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.0),
            FixedTransferProvider(0.0, 0.0),
        )
        iteration = executor.submit(0, _requests(env, 1))

        def complete_compute():
            resource = executor.resources.compute_slots[0]
            with resource.request() as request:
                yield request
                yield env.timeout(0.005)

        env.process(complete_compute())
        env.process(executor.run_iteration(iteration))
        env.run()

        a_event = next(event for event in executor.trace.events if event.stage == "a")
        self.assertAlmostEqual(a_event.ready_at, 0.0)
        self.assertAlmostEqual(a_event.start_at, 0.005)
        self.assertAlmostEqual(a_event.queue_duration, 0.005)

    def test_single_block_oracle_is_seven_milliseconds(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        compute = FixedComputeProvider(0.002, 0.003)
        transfer = FixedTransferProvider(0.001, 0.001)
        executor = _executor(env, cluster, compute, transfer)
        iteration = executor.submit(0, _requests(env, 1))

        env.process(executor.run_iteration(iteration))
        env.run()

        self.assertAlmostEqual(iteration.makespan, 0.007)
        events = executor.trace.events_for(iteration.iteration_id)
        self.assertEqual(
            [event.stage or event.operation for event in events],
            [
                "a",
                "router",
                "af_a_to_f",
                "expert",
                "af_f_to_a",
            ],
        )
        self.assertTrue(all(event.queue_duration == 0 for event in events))
        self.assertTrue(
            all(event.source_id.startswith("test-stub:") for event in events)
        )
        self.assertEqual(events[2].payload_bytes, 32)
        self.assertTrue(
            all(
                math.isclose(event.queue_duration, event.start_at - event.ready_at)
                and math.isclose(event.service_duration, event.end_at - event.start_at)
                for event in events
            )
        )
        self.assertEqual(compute.queries[0].phase, "decode")
        self.assertEqual(compute.queries[0].device_type, "a-device")
        self.assertEqual(compute.queries[1].device_type, "a-device")
        self.assertEqual(compute.queries[2].device_type, "f-device")
        self.assertEqual(transfer.queries[0].network, "af-net")
        self.assertEqual(
            [query.operation for query in transfer.queries],
            ["af_a_to_f", "af_f_to_a"],
        )

    def test_two_block_oracle_forms_eleven_millisecond_pipeline(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.003),
            FixedTransferProvider(0.001, 0.001),
        )
        iteration = executor.submit(0, _requests(env, 2))

        env.process(executor.run_iteration(iteration))
        env.run()

        self.assertAlmostEqual(iteration.makespan, 0.011)
        events = executor.trace.events_for(iteration.iteration_id)
        block_zero_expert = next(
            event for event in events if event.block_id == 0 and event.stage == "expert"
        )
        block_one_a = next(
            event for event in events if event.block_id == 1 and event.stage == "a"
        )
        self.assertLess(block_one_a.start_at, block_zero_expert.end_at)
        self.assertEqual(iteration.barrier_wait[1], 0.0)
        self.assertGreater(iteration.barrier_wait[0], 0.0)
        summary = executor.trace.summarize(iteration)
        self.assertAlmostEqual(summary["resource_utilization"]["worker:0"], 6 / 11)
        self.assertAlmostEqual(summary["resource_utilization"]["worker:1"], 4 / 11)

    def test_same_resource_group_contends_across_phase_links(self):
        makespans = {}
        for shared in (True, False):
            cluster = _two_link_cluster(shared_resource=shared)
            cluster.af.transfer_topology_mode = "flat_link"
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(0.0, 0.0),
                FixedTransferProvider(0.003, 0.003),
            )
            prefill = executor.submit(0, _requests(env, 1, phase="prefill"))
            decode = executor.submit(1, _requests(env, 1, phase="decode"))
            env.process(executor.run_iteration(prefill))
            env.process(executor.run_iteration(decode))
            env.run()
            makespans[shared] = max(prefill.end_at, decode.end_at)

        self.assertGreater(makespans[True], makespans[False])

    def test_a_and_f_bottlenecks_are_visible_in_queue_time(self):
        queue_totals = {}
        for name, a_duration, f_duration in (
            ("a-slow", 0.004, 0.001),
            ("f-slow", 0.001, 0.004),
        ):
            cluster = _load_cluster()
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(a_duration, f_duration),
                FixedTransferProvider(0.0, 0.0),
            )
            iteration = executor.submit(0, _requests(env, 3))
            env.process(executor.run_iteration(iteration))
            env.run()
            events = executor.trace.events_for(iteration.iteration_id)
            queue_totals[name] = {
                stage: sum(
                    event.queue_duration for event in events if event.stage == stage
                )
                for stage in ("a", "router", "expert")
            }

        self.assertGreater(queue_totals["a-slow"]["a"], 0)
        self.assertGreater(queue_totals["f-slow"]["expert"], 0)

    def test_multiple_layers_wait_for_previous_return(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.002, 0.003),
            FixedTransferProvider(0.001, 0.001),
            layers=2,
        )
        iteration = executor.submit(0, _requests(env, 1))

        env.process(executor.run_iteration(iteration))
        env.run()

        events = executor.trace.events_for(iteration.iteration_id)
        layer_zero_return = next(
            event
            for event in events
            if event.layer_id == 0
            and event.direction == "f_to_a"
            and event.operation == "af_f_to_a"
        )
        layer_one_a = next(
            event for event in events if event.layer_id == 1 and event.stage == "a"
        )
        self.assertEqual(layer_one_a.ready_at, layer_zero_return.end_at)
        self.assertAlmostEqual(iteration.makespan, 0.014)

    def test_full_duplex_overlaps_opposite_directions_but_half_does_not(self):
        makespans = {}
        traces = {}
        for duplex in ("full", "half"):
            cluster = _load_cluster(duplex=duplex)
            cluster.af.transfer_topology_mode = "flat_link"
            env = simpy.Environment()
            executor = _executor(
                env,
                cluster,
                FixedComputeProvider(0.001, 0.0),
                FixedTransferProvider(0.003, 0.003),
            )
            iteration = executor.submit(0, _requests(env, 2))
            env.process(executor.run_iteration(iteration))
            env.run()
            makespans[duplex] = iteration.makespan
            traces[duplex] = executor.trace.events_for(iteration.iteration_id)

        full_a_to_f = next(
            event
            for event in traces["full"]
            if event.block_id == 1
            and event.direction == "a_to_f"
            and event.operation == "af_a_to_f"
        )
        full_f_to_a = next(
            event
            for event in traces["full"]
            if event.block_id == 0
            and event.direction == "f_to_a"
            and event.operation == "af_f_to_a"
        )
        self.assertNotEqual(full_a_to_f.resource_id, full_f_to_a.resource_id)
        half_transfers = [
            event for event in traces["half"] if event.event_type == "transfer"
        ]
        service_intervals = sorted(
            (event.start_at, event.end_at) for event in half_transfers
        )
        self.assertTrue(
            all(
                left[1] <= right[0]
                for left, right in zip(service_intervals, service_intervals[1:])
            )
        )
        self.assertGreaterEqual(makespans["half"], makespans["full"])

    def test_invalid_provider_values_fail_with_context_and_release_resources(self):
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                cluster = _load_cluster()
                env = simpy.Environment()
                compute = InvalidComputeProvider(invalid)
                executor = _executor(
                    env,
                    cluster,
                    compute,
                    FixedTransferProvider(0.0, 0.0),
                )
                iteration = executor.submit(0, _requests(env, 2))
                env.process(executor.run_iteration(iteration))

                with self.assertRaisesRegex(
                    SimulationStateError, r"block=[01], layer=0, stage=a"
                ):
                    env.run()

                self.assertEqual(executor.active_iterations, {})
                self.assertTrue(
                    all(
                        resource.count == 0
                        for resource in executor.resources.compute_slots.values()
                    )
                )
                self.assertTrue(
                    all(
                        resource.count == 0
                        for resource in executor.resources.communication_resources.values()
                    )
                )
                executor.compute_provider = FixedComputeProvider(0.0, 0.0)
                recovered = executor.submit(0, _requests(env, 1))
                env.process(executor.run_iteration(recovered))
                env.run()
                self.assertEqual(recovered.makespan, 0.0)

    def test_missing_runtime_dependencies_fail_before_events(self):
        cluster = _load_cluster()
        env = simpy.Environment()
        with self.assertRaisesRegex(ConfigurationError, "compute latency provider"):
            _executor(
                env,
                cluster,
                None,
                FixedTransferProvider(0.0, 0.0),
            )

    def test_decode_only_rejects_prefill_submission(self):
        cluster = _load_cluster(phases=("decode",))
        env = simpy.Environment()
        executor = _executor(
            env,
            cluster,
            FixedComputeProvider(0.0, 0.0),
            FixedTransferProvider(0.0, 0.0),
        )
        source_worker_id = cluster.worker_groups[0].num_workers

        with self.assertRaisesRegex(ConfigurationError, "not selected"):
            executor.submit(source_worker_id, _requests(env, 1, phase="prefill"))
