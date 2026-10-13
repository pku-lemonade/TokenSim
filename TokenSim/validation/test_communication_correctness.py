"""Regression oracles for fix-af-communication-correctness."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af import AFExecutor, AFServiceEstimate, TopologyAFTransferProvider
from TokenSim.af.models import AFBlock
from TokenSim.af.planning import AFExecutionDAG, ExecutionNode, RankLocalWork
from TokenSim.config.cluster_config import AFCommunicationTimingConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware.domains import CommunicationDomainConfig
from TokenSim.hardware.placement import build_topology_placement
from tests.test_af_executor import af_test_hardware
from tests.test_af_operator_provider import _transfer_query
from tests.test_af_parallel_planning import _cluster


class InputCompute:
    def __init__(self, late=4, delay=100e-6, fail=False):
        self.late, self.delay, self.fail = late, delay, fail

    def estimate(self, query):
        if self.fail and query.worker_id == self.late:
            raise ValueError('producer failed')
        return AFServiceEstimate(self.delay if query.worker_id == self.late else 0,
                                 'test:input', details={'timing_scope': 'device_service'})


def fixture(*, flat=False, cache=True, timing=False, compute=None, link_parameters=None):
    cluster = _cluster(a_tp=8, a_dp=1, f_tp=8, f_dp=1)
    cluster.af.communication_domains = CommunicationDomainConfig(node_size=8, node_count=4,
        local_latency_us=1, inter_latency_us=2,
        device_indices={f'{pool}:{rank}': offset + (rank // 4) * 8 + rank % 4
                        for pool, offset in (('a-pool', 0), ('f-pool', 16)) for rank in range(8)})
    cluster.af.transfer_topology_mode = 'flat_link' if flat else 'physical'
    cluster.af.service_estimate_cache = cache
    cluster.af.communication_timing = AFCommunicationTimingConfig(enabled=timing,
        host_submit_us=1, host_completion_us=1)
    cluster.af.trace_enabled = True
    workers = cluster.workers()
    hardware = af_test_hardware()
    if link_parameters:
        from TokenSim.hardware.links import LinkCatalog
        hardware.links = LinkCatalog(replace(link,**link_parameters) for link in hardware.links)
    placement = build_topology_placement(cluster, workers, hardware)
    provider = TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only')
    executor = AFExecutor(simpy.Environment(), cluster, workers, compute or InputCompute(), provider,
                          'TestModel', 'bf16', 1, 16)
    return executor, provider, hardware, workers, placement


def dag_for(executor, operation='all_reduce', ranks=tuple(range(8))):
    identities = tuple(executor.resources.worker(rank).physical_worker_ref.identity for rank in ranks)
    common = dict(source_domain='a-domain', target_domain='a-domain', logical_participants=('a-domain:dp0',), dp_rank=0)
    producers = tuple(ExecutionNode(node_id=f'p:{rank}', operation='attention', kind='compute',
        predecessor_ids=(), stage='a', direction=None, domain_ranks=(rank,), physical_workers=(identity,),
        local_work=RankLocalWork(1, (('tokens',1),('hidden_size',16),('tp_size',1))), **common)
        for rank, identity in zip(ranks, identities))
    extra = {'input_producer_ids': tuple(zip(identities, (node.node_id for node in producers)))} \
        if 'input_producer_ids' in ExecutionNode.__dataclass_fields__ else {}
    collective = ExecutionNode(node_id='test:collective', operation=operation, kind='communication',
        predecessor_ids=tuple(node.node_id for node in producers), stage=None, direction='collective',
        domain_ranks=ranks, physical_workers=identities, payload_bytes=800,
        resource_lane='a_to_f', network=executor.cluster_config.af.links[0].network, logical_stage_id='test:collective', **common, **extra)
    consumer = replace(producers[0], node_id='test:consumer', predecessor_ids=('test:collective',))
    return AFExecutionDAG((*producers, collective, consumer))


def run_dag(executor, dag, iteration=1):
    block = AFBlock(iteration, 0, 'prefill', (0,), 1, 1, (1,), 0, 'af-link')
    process = executor.env.process(executor._run_dag(block, -1,
        executor.cluster_config.af.links[0], dag))
    executor.env.run()
    return process


def measured_record(provider, query, **overrides):
    phase = provider.communication_plan(query).phases[0]
    # analytical details contain a timetable, while measured signatures use node-normalized edges.
    nodes = {node: i for i,node in enumerate(sorted({provider.node_by_worker[r] for r in phase.participants}))}
    signature = dict(model_version='node-domain-fifo-v1', algorithm='direct',
        phase_kind='transfer', operation='send_recv', dtype=phase.dtype, participants=len(phase.participants),
        nodes=len(nodes), logical_input_bytes=phase.logical_input_bytes, wire_bytes=phase.wire_bytes,
        wire_schedule=tuple(tuple((nodes[m.source_node],nodes[m.target_node],m.wire_bytes) for m in row)
                            for row in phase.rounds),
        domain_parameters=tuple((name,getattr(provider.domain_config,name)) for name in (
            'local_bandwidth_bytes_per_s','inter_tx_bandwidth_bytes_per_s','inter_rx_bandwidth_bytes_per_s',
            'local_latency_us','inter_latency_us')),
        timing_scope='transport_service', latency_us=7, source_id='test:A-only')
    return {**signature, **overrides}


class CommunicationBugReproductions(unittest.TestCase):
    def test_ready_subgroup_starts_at_zero(self):
        executor, *_ = fixture()
        run_dag(executor, dag_for(executor))
        event = next(e for e in executor.trace.events if e.details.get('phase_id') == 'local_rs:0')
        self.assertEqual(event.start_at, 0)

    def test_a_measurement_does_not_price_f_device(self):
        executor, base, hardware, workers, placement = fixture()
        query = replace(_transfer_query(payload_bytes=100, endpoints=('a-pool:0','a-pool:1')), dtype='bf16')
        record = measured_record(base, query)
        package = SimpleNamespace(meta=SimpleNamespace(dataset_version='test-v1',
            extra={'communication_phase_services': (record,)}))
        with patch.object(hardware, 'operator_package', side_effect=lambda device, backend:
                          package if device == 'a-device' else None):
            provider = TopologyAFTransferProvider(hardware, executor.cluster_config, workers, placement)
        fquery = replace(query, physical_endpoints=('f-pool:0','f-pool:1'))
        self.assertEqual(provider.estimate(fquery).match_type, 'analytical')

    def test_flat_rejects_bad_fraction(self):
        _, provider, *_ = fixture(flat=True)
        query = replace(_transfer_query('all_to_all', payload_bytes=100,
            endpoints=('a-pool:0','a-pool:1')), remote_fraction=-1)
        with self.assertRaises(ConfigurationError):
            provider.estimate(query)

    def test_flat_checks_dtype_alignment(self):
        _, provider, *_ = fixture(flat=True)
        query = replace(_transfer_query('all_to_all', payload_bytes=3,
            endpoints=('a-pool:0','a-pool:1')), dtype='fp16')
        with self.assertRaises(ConfigurationError):
            provider.estimate(query)


class SubgroupReadinessTests(unittest.TestCase):
    def test_completion_and_phase_dependencies_with_host_and_cache(self):
        traces = []
        for cache in (False, True):
            for timing in (False, True):
                executor, *_ = fixture(cache=cache, timing=timing)
                run_dag(executor, dag_for(executor))
                events = executor.trace.events
                phases = {e.details['phase_id']: e for e in events if 'phase_id' in e.details}
                self.assertLess(phases['local_rs:0'].end_at, 100e-6)
                self.assertGreaterEqual(phases['local_rs:1'].start_at, 100e-6)
                for slot in range(4):
                    self.assertGreaterEqual(phases[f'inter_ar:{slot}'].start_at,
                        max(phases['local_rs:0'].end_at, phases['local_rs:1'].end_at))
                completion = next(e for e in events if e.node_id == ('test:collective:transport' if timing else 'test:collective'))
                consumer = next(e for e in events if e.node_id == 'test:consumer')
                self.assertGreaterEqual(consumer.start_at, completion.end_at)
                if timing:
                    submit = [e for e in events if e.event_type == 'host_submit']
                    self.assertAlmostEqual(sum(e.service_duration for e in submit), 8e-6)
                    self.assertTrue(all(e.start_at == 0 for e in submit))
                executor.trace.validate_invariants(1)
                traces.append((cache, timing, [(e.node_id,e.ready_at,e.start_at,e.end_at,e.predecessor_ids) for e in events]))
        self.assertEqual(traces[0][2], traces[2][2])
        self.assertEqual(traces[1][2], traces[3][2])

    def test_allgather_slot_and_reduce_scatter_reorder(self):
        for operation, ranks, early_id, late_id in (
            ('all_gather',tuple(range(8)),'inter_ag:1','inter_ag:0'),
            ('reduce_scatter',(0,4,1,5,2,6,3,7),'input_layout:0','input_layout:1')):
            executor, *_ = fixture()
            run_dag(executor, dag_for(executor,operation,ranks))
            phases = {e.details['phase_id']:e for e in executor.trace.events if 'phase_id' in e.details}
            # Layout phase is a completion marker; its per-worker compute children begin at readiness.
            if operation == 'all_gather':
                self.assertEqual(phases[early_id].start_at,0)
                self.assertGreaterEqual(phases[late_id].start_at,100e-6)
            else:
                early = [e for e in executor.trace.events if e.node_id and f':phase:{early_id}:worker:' in e.node_id]
                late = [e for e in executor.trace.events if e.node_id and f':phase:{late_id}:worker:' in e.node_id]
                self.assertEqual(min(e.start_at for e in early),0)
                self.assertGreaterEqual(min(e.start_at for e in late),100e-6)
            executor.trace.validate_invariants(1)

    def test_dependency_wait_is_not_resource_queue(self):
        executor, *_ = fixture(compute=InputCompute(late=0,delay=10e-6))
        # Force local node 0 to remain busy through t=16 us.
        executor.env.process(executor.resources.domain_scheduler.run(('domain:node0:local',),16e-6,('busy',)))
        run_dag(executor,dag_for(executor))
        phase = next(e for e in executor.trace.events if e.details.get('phase_id') == 'local_rs:0')
        self.assertAlmostEqual(phase.ready_at,10e-6)
        self.assertAlmostEqual(phase.start_at,16e-6)
        self.assertAlmostEqual(phase.queue_duration,6e-6)
        self.assertAlmostEqual(phase.end_at-phase.start_at,phase.service_duration)

    def test_binding_validation_and_cycle(self):
        executor, *_ = fixture()
        dag = dag_for(executor)
        node = dag.nodes[-2]
        for bindings in (node.input_producer_ids[:-1],
                         (node.input_producer_ids[0],)*8,
                         tuple(reversed(node.input_producer_ids)),
                         ((node.physical_workers[0],'missing'),)+node.input_producer_ids[1:]):
            with self.assertRaises(ConfigurationError):
                replace(node,input_producer_ids=bindings)
        with self.assertRaises(ConfigurationError):
            replace(node,input_producer_ids=(),metadata={'collective_input_binding_required':True})
        cyclic = replace(node,predecessor_ids=(*node.predecessor_ids,'test:consumer'),
            input_producer_ids=((node.physical_workers[0],'test:consumer'),)+node.input_producer_ids[1:])
        with self.assertRaisesRegex(ConfigurationError,'cyclic'):
            AFExecutionDAG((*dag.nodes[:-2],cyclic,dag.nodes[-1]))

    def test_legacy_and_flat_collectives_keep_barrier(self):
        for flat, legacy in ((True,False),(False,True)):
            executor, *_ = fixture(flat=flat)
            dag = dag_for(executor)
            if legacy:
                dag = AFExecutionDAG((*dag.nodes[:-2],replace(dag.nodes[-2],input_producer_ids=()),dag.nodes[-1]))
            run_dag(executor,dag)
            service = [e for e in executor.trace.events if e.operation in {'all_reduce','reduce_scatter'}]
            self.assertTrue(all(e.start_at >= 100e-6 for e in service))

    def test_invocations_and_template_binding_have_new_events(self):
        from TokenSim.af.templates import DAGTemplate
        executor, provider, *_ = fixture()
        dag = dag_for(executor)
        first = AFBlock(1,0,'prefill',(0,),1,1,(1,),0,'af-link')
        names = {node.node_id:f'i1:b0:l-1:{node.node_id}' for node in dag.nodes}
        dag = AFExecutionDAG(tuple(replace(node,node_id=names[node.node_id],
            predecessor_ids=tuple(names[key] for key in node.predecessor_ids),
            input_producer_ids=tuple((endpoint,names[key]) for endpoint,key in node.input_producer_ids)) for node in dag.nodes))
        template = DAGTemplate.compile(dag,first,-1,(),())
        run_dag(executor,template.bind(first,-1,(),()))
        origin = executor.env.now
        executor.compute_provider = InputCompute(late=0,delay=200e-6)
        second = replace(first,iteration_id=2)
        run_dag(executor,template.bind(second,-1,(),()),iteration=2)
        phase = next(e for e in executor.trace.events_for(2) if e.details.get('phase_id') == 'local_rs:0')
        self.assertGreaterEqual(phase.ready_at,origin+200e-6)
        self.assertTrue(all(key.startswith('i2:') for key in phase.predecessor_ids))
        self.assertGreater(provider.plan_cache_hits,0)
        self.assertFalse(executor.resources.domain_scheduler.owners)

    def test_failure_and_cancellation_release_waiters(self):
        for failure in ('producer','phase','cancel'):
            executor, provider, *_ = fixture()
            dag = dag_for(executor)
            env = executor.env
            original = executor._run_planned_compute
            def delayed_failure(block,layer,step):
                if step.node_id == 'p:4':
                    yield env.timeout(2e-6)
                    raise ValueError('late producer failure')
                yield from original(block,layer,step)
            if failure == 'producer':
                executor._run_planned_compute = delayed_failure
            if failure == 'phase':
                original_phase = provider.phase_estimate
                calls = {}
                def broken(phase):
                    calls[phase.phase_id] = calls.get(phase.phase_id,0)+1
                    if phase.phase_id == 'local_rs:1' and calls[phase.phase_id] > 1:
                        raise ValueError('phase failure')
                    return original_phase(phase)
                provider.phase_estimate = broken
            block = AFBlock(1,0,'prefill',(0,),1,1,(1,),0,'af-link')
            operation = env.process(executor._run_dag(block,-1,executor.cluster_config.af.links[0],dag))
            if failure == 'cancel':
                def cancel():
                    yield env.timeout(2e-6)
                    operation.interrupt('cancel')
                env.process(cancel())
            with self.assertRaises((SimulationStateError,simpy.Interrupt)):
                env.run()
            env.run()
            self.assertFalse(any(e.node_id=='test:consumer' for e in executor.trace.events))
            scheduler = executor.resources.domain_scheduler
            self.assertFalse(scheduler.owners)
            self.assertFalse(scheduler.waiters)
            self.assertFalse(executor.active_stages)
            following = env.process(scheduler.run(('domain:node0:local',),1e-6,('following',)))
            env.run()
            self.assertTrue(following.triggered)


class MeasurementApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.executor, self.base, self.hardware, self.workers, self.placement = fixture()
        self.query = replace(_transfer_query(payload_bytes=100,endpoints=('a-pool:0','a-pool:1')),dtype='bf16')

    def provider(self, records, *, backend=None, fallback='table_first', workers=None):
        package = SimpleNamespace(meta=SimpleNamespace(dataset_version='test-v1',
            extra={'communication_phase_services':tuple(records)}))
        with patch.object(self.hardware,'operator_package',side_effect=lambda device, package_backend:
                          package if device=='a-device' else None):
            return TopologyAFTransferProvider(self.hardware,self.executor.cluster_config,
                workers or self.workers,self.placement,operator_backend=backend,fallback=fallback)

    def test_matching_scope_backend_and_missing_fallback(self):
        record = measured_record(self.base,self.query,transport_backend='nccl')
        provider = self.provider((record,))
        measured = provider.estimate(replace(self.query,transport_backend='nccl'))
        self.assertEqual(measured.match_type,'exact')
        self.assertAlmostEqual(measured.service_duration,7e-6)
        phase = provider.communication_plan(replace(self.query,transport_backend='nccl')).phases[0]
        details = provider.phase_estimate(phase).details
        self.assertEqual(details['participant_hardware_backends'],(('a-device',None),)*2)
        self.assertEqual(details['measurement_device'],'a-device')
        self.assertEqual(provider.estimate(replace(self.query,transport_backend='deepep')).match_type,'analytical')
        fquery = replace(self.query,physical_endpoints=('f-pool:0','f-pool:1'))
        for query in (self.query,fquery):
            with self.assertRaisesRegex(ConfigurationError,'matching'):
                self.provider((record,),fallback='table_only').estimate(query)

    def test_homogeneous_aliases_and_explicit_mixed_scope(self):
        record = measured_record(self.base,self.query)
        # F role using the same actual accelerator is allowed; role identity is not hardware identity.
        workers = [replace(w,hardware='a-device') if w.hardware=='f-device' else w for w in self.workers]
        provider = self.provider((record,),workers=workers)
        self.assertEqual(provider.estimate(replace(self.query,physical_endpoints=('f-pool:0','f-pool:1'))).match_type,'exact')
        mixed = replace(self.query,physical_endpoints=('a-pool:0','f-pool:0'))
        mixed_record = measured_record(self.base,mixed)
        self.assertEqual(self.provider((mixed_record,)).estimate(mixed).match_type,'analytical')
        explicit = {**mixed_record,'participant_hardware_backends':(('a-device',None),('f-device',None))}
        self.assertEqual(self.provider((explicit,)).estimate(mixed).match_type,'exact')
        self.assertEqual(self.provider((explicit,)).estimate(replace(mixed,physical_endpoints=tuple(reversed(mixed.physical_endpoints)))).match_type,'analytical')

    def test_package_backend_scope_dedup_and_conflict(self):
        record = measured_record(self.base,self.query,
            participant_hardware_backends=(('a-device','backend-a'),)*2)
        self.assertEqual(self.provider((record,),backend='backend-b').estimate(self.query).match_type,'analytical')
        provider = self.provider((record,record),backend='backend-a')
        self.assertEqual(len(provider.phase_measurements),1)
        self.assertEqual(provider.estimate(self.query).match_type,'exact')
        with self.assertRaisesRegex(ConfigurationError,'conflicting'):
            self.provider((record,{**record,'latency_us':9,'source_id':'test:other'}),backend='backend-a').estimate(self.query)

    def test_scope_and_measurement_content_cache_equivalence(self):
        record = measured_record(self.base,self.query)
        results = []
        for latency in (7,9):
            for backend in ('backend-a','backend-b'):
                provider = self.provider(({**record,'latency_us':latency},),backend=backend)
                for cache in (True,False):
                    provider.cache_enabled = cache
                    result = provider.estimate(self.query)
                    self.assertAlmostEqual(result.service_duration,latency*1e-6)
                    results.append((latency,backend,result))
        for index in range(0,len(results),2):
            self.assertEqual(results[index],results[index+1])

    def test_actual_package_backend_and_device_aliases(self):
        record = measured_record(self.base,self.query)
        package = SimpleNamespace(meta=SimpleNamespace(dataset_version='test-v1',backend='actual-backend',
            extra={'communication_phase_services':(record,)}))
        original_device = self.hardware.device
        with patch.object(self.hardware,'device',side_effect=lambda name:
                          original_device('a-device' if name=='f-device' else name)), \
             patch.object(self.hardware,'operator_package',return_value=package):
            provider = TopologyAFTransferProvider(self.hardware,self.executor.cluster_config,self.workers,self.placement)
        result = provider.estimate(replace(self.query,physical_endpoints=('f-pool:0','f-pool:1')))
        self.assertEqual(result.match_type,'exact')
        phase = provider.communication_plan(self.query).phases[0]
        details = provider.phase_estimate(phase).details
        self.assertEqual(details['measurement_backend'],'actual-backend')
        self.assertEqual(details['participant_hardware_backends'],(('a-device','actual-backend'),)*2)


class AllToAllContractTests(unittest.TestCase):
    def test_invalid_fields_both_modes_and_noops(self):
        bad = [dict(payload_bytes=-1),dict(payload_bytes=True),dict(payload_bytes=3,dtype='fp16'),
            dict(payload_bytes=2.0),dict(local_fraction=True),dict(local_fraction=float('nan')),
            dict(remote_fraction=float('inf')),dict(remote_fraction=-1),
            dict(local_fraction=.3,remote_fraction=.6),dict(load_imbalance=.9),
            dict(load_imbalance=True),dict(load_imbalance=float('nan')),
            dict(load_imbalance=float('inf')),dict(destination_redistribution=1),
            dict(group_size=3),dict(group_size=True),dict(physical_endpoints=()),
            dict(physical_endpoints=('a-pool:0','a-pool:0')),dict(physical_endpoints=('missing',)),
            dict(participant_ranks=(0,0))]
        for flat in (False,True):
            _, provider, *_ = fixture(flat=flat)
            for size,endpoints in ((0,('a-pool:0',)),(100,('a-pool:0','a-pool:1'))):
                query = _transfer_query('all_to_all',payload_bytes=size,endpoints=endpoints)
                query = replace(query,group_size=len(endpoints))
                for fields in bad:
                    with self.subTest(flat=flat,size=size,fields=fields), self.assertRaises(ConfigurationError):
                        provider.estimate(replace(query,**fields))

    def test_uniform_formula_support_rejection_and_cache(self):
        for cache in (True,False):
            executor, provider, *_ = fixture(flat=True,cache=cache,
                link_parameters=dict(bandwidth_per_direction_bytes_per_s=50e9,latency_us=2))
            query = replace(_transfer_query('all_to_all',payload_bytes=400,
                endpoints=tuple(f'a-pool:{r}' for r in range(4))),group_size=4)
            link = provider.link_class(query.network)
            estimate = provider.estimate(query)
            self.assertAlmostEqual(estimate.service_duration,(link.latency_us+300/link.bandwidth_per_direction_bytes_per_s*1e6)/1e6)
            self.assertAlmostEqual(estimate.service_duration,2.006e-6)
            self.assertEqual(estimate.details['wire_bytes'],300)
            self.assertEqual(estimate.details['wire_bytes_scope'],'per_rank')
            for fields in (dict(remote_fraction=.75),dict(local_fraction=1),dict(load_imbalance=2),
                           dict(destination_redistribution=True)):
                with self.assertRaisesRegex(ConfigurationError,'physical'):
                    provider.estimate(replace(query,**fields))
            for size,count in ((0,4),(400,1)):
                q = replace(query,payload_bytes=size,physical_endpoints=query.physical_endpoints[:count],group_size=count,
                            participant_ranks=tuple(range(count)))
                self.assertEqual(provider.estimate(q).service_duration,0)
                node = dag_for(executor).nodes[-2]
                node = replace(node,operation='all_to_all',domain_ranks=tuple(range(count)),
                    physical_workers=q.physical_endpoints,payload_bytes=size,input_producer_ids=(),predecessor_ids=())
                requested = []
                for _,resource in executor.resources.communication_resources.items():
                    original = resource.request
                    def spy(original=original):
                        requested.append(True)
                        return original()
                    resource.request = spy
                block = AFBlock(99,0,'prefill',(0,),1,1,(1,),0,'af-link')
                executor.env.process(executor._run_planned_communication(block,0,executor.cluster_config.af.links[0],node))
                executor.env.run()
                self.assertFalse(requested)

    def test_physical_statistics_still_conserve_outputs(self):
        _, provider, *_ = fixture()
        query = replace(_transfer_query('all_to_all',payload_bytes=400,
            endpoints=tuple(f'a-pool:{r}' for r in range(8))),group_size=8,
            remote_fraction=.75,load_imbalance=2,destination_redistribution=True)
        plan = provider.communication_plan(query)
        self.assertEqual(sum(size for _,size in plan.expected_output_bytes),8*400)
        self.assertEqual(plan.provenance,'statistical_locality')
        self.assertTrue(any(phase.phase_id.startswith('destination_local:') for phase in plan.phases))


if __name__ == '__main__':
    unittest.main()
