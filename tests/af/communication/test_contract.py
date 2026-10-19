"""Communication resource, volume and readiness regression contracts."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.providers.operator_table import TopologyAFTransferProvider
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.af.communication import AFCommunicationTimingConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware.domains import CommunicationDomainConfig
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.config import _cluster
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _transfer_query


class InputCompute:
    def __init__(self, late=4, delay=100e-6, fail=False):
        self.late, self.delay, self.fail = late, delay, fail

    def estimate(self, query):
        if self.fail and query.worker_id == self.late:
            raise ValueError("producer failed")
        return AFServiceEstimate(
            self.delay if query.worker_id == self.late else 0,
            "test:input",
            details={"timing_scope": "device_service"},
        )


def fixture(
    *, flat=False, cache=True, timing=False, compute=None, link_parameters=None
):
    cluster = _cluster(a_tp=8, a_dp=1, f_tp=8, f_dp=1)
    cluster.af.communication_domains = CommunicationDomainConfig(
        node_size=8,
        node_count=4,
        local_latency_us=1,
        inter_latency_us=2,
        device_indices={
            f"{pool}:{rank}": offset + (rank // 4) * 8 + rank % 4
            for pool, offset in (("a-pool", 0), ("f-pool", 16))
            for rank in range(8)
        },
    )
    cluster.af.transfer_topology_mode = "flat_link" if flat else "physical"
    cluster.af.service_estimate_cache = cache
    cluster.af.communication_timing = AFCommunicationTimingConfig(
        enabled=timing, host_submit_us=1, host_completion_us=1
    )
    cluster.af.trace_enabled = True
    workers = cluster.workers()
    hardware = af_test_hardware()
    if link_parameters:
        from TokenSim.hardware.links import LinkCatalog

        hardware.links = LinkCatalog(
            replace(link, **link_parameters) for link in hardware.links
        )
    placement = build_topology_placement(cluster, workers, hardware)
    provider = TopologyAFTransferProvider(
        hardware, cluster, workers, placement, fallback="analytical_only"
    )
    executor = AFExecutor(
        simpy.Environment(),
        cluster,
        workers,
        compute or InputCompute(),
        provider,
        "TestModel",
        "bf16",
        1,
        16,
    )
    return executor, provider, hardware, workers, placement


def dag_for(executor, operation="all_reduce", ranks=tuple(range(8))):
    identities = tuple(
        executor.resources.worker(rank).physical_worker_ref.identity for rank in ranks
    )
    common = dict(
        source_domain="a-domain",
        target_domain="a-domain",
        logical_participants=("a-domain:dp0",),
        dp_rank=0,
    )
    producers = tuple(
        ExecutionNode(
            node_id=f"p:{rank}",
            operation="attention",
            kind="compute",
            predecessor_ids=(),
            stage="a",
            direction=None,
            domain_ranks=(rank,),
            physical_workers=(identity,),
            local_work=RankLocalWork(
                1, (("tokens", 1), ("hidden_size", 16), ("tp_size", 1))
            ),
            **common,
        )
        for rank, identity in zip(ranks, identities)
    )
    extra = (
        {
            "input_producer_ids": tuple(
                zip(identities, (node.node_id for node in producers))
            )
        }
        if "input_producer_ids" in ExecutionNode.__dataclass_fields__
        else {}
    )
    collective = ExecutionNode(
        node_id="test:collective",
        operation=operation,
        kind="communication",
        predecessor_ids=tuple(node.node_id for node in producers),
        stage=None,
        direction="collective",
        domain_ranks=ranks,
        physical_workers=identities,
        payload_bytes=800,
        resource_lane="a_to_f",
        network=executor.cluster_config.af.links[0].network,
        logical_stage_id="test:collective",
        **common,
        **extra,
    )
    consumer = replace(
        producers[0], node_id="test:consumer", predecessor_ids=("test:collective",)
    )
    return AFExecutionDAG((*producers, collective, consumer))


def run_dag(executor, dag, iteration=1):
    block = AFBlock(iteration, 0, (0,), 1, 1, (1,), 0, "af-link")
    process = executor.env.process(
        executor._run_dag(block, -1, executor.cluster_config.af.links[0], dag)
    )
    executor.env.run()
    return process


def measured_record(provider, query, **overrides):
    phase = provider.communication_plan(query).phases[0]
    # analytical details contain a timetable, while measured signatures use node-normalized edges.
    nodes = {
        node: i
        for i, node in enumerate(
            sorted({provider.node_by_worker[r] for r in phase.participants})
        )
    }
    signature = dict(
        model_version="node-domain-fifo-v1",
        algorithm="direct",
        phase_kind="transfer",
        operation="send_recv",
        dtype=phase.dtype,
        participants=len(phase.participants),
        nodes=len(nodes),
        logical_input_bytes=phase.logical_input_bytes,
        wire_bytes=phase.wire_bytes,
        wire_schedule=tuple(
            tuple(
                (nodes[m.source_node], nodes[m.target_node], m.wire_bytes) for m in row
            )
            for row in phase.rounds
        ),
        domain_parameters=tuple(
            (name, getattr(provider.domain_config, name))
            for name in (
                "local_bandwidth_bytes_per_s",
                "inter_tx_bandwidth_bytes_per_s",
                "inter_rx_bandwidth_bytes_per_s",
                "local_latency_us",
                "inter_latency_us",
            )
        ),
        timing_scope="transport_service",
        latency_us=7,
        source_id="test:A-only",
    )
    return {**signature, **overrides}


class SubgroupReadinessTests(unittest.TestCase):
    def test_completion_and_phase_dependencies_with_host_and_cache(self):
        traces = []
        for cache in (False, True):
            for timing in (False, True):
                executor, *_ = fixture(cache=cache, timing=timing)
                run_dag(executor, dag_for(executor))
                events = executor.trace.events
                phases = {
                    e.details["phase_id"]: e for e in events if "phase_id" in e.details
                }
                self.assertLess(phases["local_rs:0"].end_at, 100e-6)
                self.assertGreaterEqual(phases["local_rs:1"].start_at, 100e-6)
                for slot in range(4):
                    self.assertGreaterEqual(
                        phases[f"inter_ar:{slot}"].start_at,
                        max(phases["local_rs:0"].end_at, phases["local_rs:1"].end_at),
                    )
                completion = next(
                    e
                    for e in events
                    if e.node_id
                    == ("test:collective:transport" if timing else "test:collective")
                )
                consumer = next(e for e in events if e.node_id == "test:consumer")
                self.assertGreaterEqual(consumer.start_at, completion.end_at)
                if timing:
                    submit = [e for e in events if e.event_type == "host_submit"]
                    self.assertAlmostEqual(
                        sum(e.service_duration for e in submit), 8e-6
                    )
                    self.assertTrue(all(e.start_at == 0 for e in submit))
                executor.trace.validate_invariants(1)
                traces.append(
                    (
                        cache,
                        timing,
                        [
                            (
                                e.node_id,
                                e.ready_at,
                                e.start_at,
                                e.end_at,
                                e.predecessor_ids,
                            )
                            for e in events
                        ],
                    )
                )
        self.assertEqual(traces[0][2], traces[2][2])
        self.assertEqual(traces[1][2], traces[3][2])

    def test_dependency_wait_is_not_resource_queue(self):
        executor, *_ = fixture(compute=InputCompute(late=0, delay=10e-6))
        # Force local node 0 to remain busy through t=16 us.
        executor.env.process(
            executor.resources.domain_scheduler.run(
                ("domain:node0:local",), 16e-6, ("busy",)
            )
        )
        run_dag(executor, dag_for(executor))
        phase = next(
            e
            for e in executor.trace.events
            if e.details.get("phase_id") == "local_rs:0"
        )
        self.assertAlmostEqual(phase.ready_at, 10e-6)
        self.assertAlmostEqual(phase.start_at, 16e-6)
        self.assertAlmostEqual(phase.queue_duration, 6e-6)
        self.assertAlmostEqual(phase.end_at - phase.start_at, phase.service_duration)

    def test_failure_and_cancellation_release_waiters(self):
        for failure in ("producer", "phase", "cancel"):
            executor, provider, *_ = fixture()
            dag = dag_for(executor)
            env = executor.env
            original = executor._run_planned_compute

            def delayed_failure(block, layer, step):
                if step.node_id == "p:4":
                    yield env.timeout(2e-6)
                    raise ValueError("late producer failure")
                yield from original(block, layer, step)

            if failure == "producer":
                executor._run_planned_compute = delayed_failure
            if failure == "phase":
                original_phase = provider.phase_estimate
                calls = {}

                def broken(phase):
                    calls[phase.phase_id] = calls.get(phase.phase_id, 0) + 1
                    if phase.phase_id == "local_rs:1" and calls[phase.phase_id] > 1:
                        raise ValueError("phase failure")
                    return original_phase(phase)

                provider.phase_estimate = broken
            block = AFBlock(1, 0, (0,), 1, 1, (1,), 0, "af-link")
            operation = env.process(
                executor._run_dag(block, -1, executor.cluster_config.af.links[0], dag)
            )
            if failure == "cancel":

                def cancel():
                    yield env.timeout(2e-6)
                    operation.interrupt("cancel")

                env.process(cancel())
            with self.assertRaises((SimulationStateError, simpy.Interrupt)):
                env.run()
            env.run()
            self.assertFalse(
                any(e.node_id == "test:consumer" for e in executor.trace.events)
            )
            scheduler = executor.resources.domain_scheduler
            self.assertFalse(scheduler.owners)
            self.assertFalse(scheduler.waiters)
            self.assertFalse(executor.active_stages)
            following = env.process(
                scheduler.run(("domain:node0:local",), 1e-6, ("following",))
            )
            env.run()
            self.assertTrue(following.triggered)


class MeasurementApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.executor, self.base, self.hardware, self.workers, self.placement = (
            fixture()
        )
        self.query = replace(
            _transfer_query(payload_bytes=100, endpoints=("a-pool:0", "a-pool:1")),
            dtype="bf16",
        )

    def provider(self, records, *, backend=None, fallback="table_first", workers=None):
        package = SimpleNamespace(
            meta=SimpleNamespace(
                dataset_version="test-v1",
                extra={"communication_phase_services": tuple(records)},
            )
        )
        with patch.object(
            self.hardware,
            "operator_package",
            side_effect=lambda device, package_backend: (
                package if device == "a-device" else None
            ),
        ):
            return TopologyAFTransferProvider(
                self.hardware,
                self.executor.cluster_config,
                workers or self.workers,
                self.placement,
                operator_backend=backend,
                fallback=fallback,
            )

    def test_matching_scope_backend_and_missing_fallback(self):
        record = measured_record(self.base, self.query, transport_backend="nccl")
        provider = self.provider((record,))
        measured = provider.estimate(replace(self.query, transport_backend="nccl"))
        self.assertEqual(measured.match_type, "exact")
        self.assertAlmostEqual(measured.service_duration, 7e-6)
        phase = provider.communication_plan(
            replace(self.query, transport_backend="nccl")
        ).phases[0]
        details = provider.phase_estimate(phase).details
        self.assertEqual(
            details["participant_hardware_backends"], (("a-device", None),) * 2
        )
        self.assertEqual(details["measurement_device"], "a-device")
        self.assertEqual(
            provider.estimate(
                replace(self.query, transport_backend="deepep")
            ).match_type,
            "analytical",
        )
        fquery = replace(self.query, physical_endpoints=("f-pool:0", "f-pool:1"))
        for query in (self.query, fquery):
            with self.assertRaisesRegex(ConfigurationError, "matching"):
                self.provider((record,), fallback="table_only").estimate(query)


class AllToAllContractTests(unittest.TestCase):
    def test_invalid_fields_both_modes_and_noops(self):
        bad = [
            dict(payload_bytes=-1),
            dict(payload_bytes=True),
            dict(payload_bytes=3, dtype="fp16"),
            dict(payload_bytes=2.0),
            dict(local_fraction=True),
            dict(local_fraction=float("nan")),
            dict(remote_fraction=float("inf")),
            dict(remote_fraction=-1),
            dict(local_fraction=0.3, remote_fraction=0.6),
            dict(load_imbalance=0.9),
            dict(load_imbalance=True),
            dict(load_imbalance=float("nan")),
            dict(load_imbalance=float("inf")),
            dict(destination_redistribution=1),
            dict(group_size=3),
            dict(group_size=True),
            dict(physical_endpoints=()),
            dict(physical_endpoints=("a-pool:0", "a-pool:0")),
            dict(physical_endpoints=("missing",)),
            dict(participant_ranks=(0, 0)),
        ]
        for flat in (False, True):
            _, provider, *_ = fixture(flat=flat)
            for size, endpoints in (
                (0, ("a-pool:0",)),
                (100, ("a-pool:0", "a-pool:1")),
            ):
                query = _transfer_query(
                    "all_to_all", payload_bytes=size, endpoints=endpoints
                )
                query = replace(query, group_size=len(endpoints))
                for fields in bad:
                    with (
                        self.subTest(flat=flat, size=size, fields=fields),
                        self.assertRaises(ConfigurationError),
                    ):
                        provider.estimate(replace(query, **fields))

    def test_physical_statistics_still_conserve_outputs(self):
        _, provider, *_ = fixture()
        query = replace(
            _transfer_query(
                "all_to_all",
                payload_bytes=400,
                endpoints=tuple(f"a-pool:{r}" for r in range(8)),
            ),
            group_size=8,
            remote_fraction=0.75,
            load_imbalance=2,
            destination_redistribution=True,
        )
        plan = provider.communication_plan(query)
        self.assertEqual(sum(size for _, size in plan.expected_output_bytes), 8 * 400)
        self.assertEqual(plan.provenance, "statistical_locality")
        self.assertTrue(
            any(
                phase.phase_id.startswith("destination_local:") for phase in plan.phases
            )
        )
