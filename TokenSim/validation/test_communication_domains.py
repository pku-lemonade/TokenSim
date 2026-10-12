"""Behavioral oracles for node budgets, phase volumes, FIFO and cache reuse."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.domain_scheduler import DomainScheduler
from TokenSim.comm.traffic import collective_plan, all_to_all_plan, ring_volume
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig, transmission_service
from TokenSim.hardware.links import LinkClass


class VolumeTest(unittest.TestCase):
    def test_numeric_tensor_identity_across_hierarchical_phases(self):
        from TokenSim.validation.collective_oracle import execute_hierarchy
        for nodes in ({rank:0 for rank in range(4)}, {rank:rank//4 for rank in range(8)},
                      {rank:rank%2 for rank in range(8)}):
            ranks = tuple(nodes)
            for operation in ('all_gather','all_reduce','reduce_scatter'):
                inputs = {rank:[rank*100+index for index in range(16)] for rank in ranks}
                plan = collective_plan(operation, ranks, 32, nodes)
                output = execute_hierarchy(plan, inputs)
                if operation == 'all_gather':
                    expected = [value for rank in ranks for value in inputs[rank]]
                    self.assertTrue(all(value == expected for value in output.values()))
                else:
                    expected = [sum(inputs[rank][index] for rank in ranks) for index in range(16)]
                    if operation == 'all_reduce':
                        self.assertTrue(all(value == expected for value in output.values()))
                    else:
                        chunk = 16//len(ranks)
                        self.assertEqual(output,{rank:expected[index*chunk:(index+1)*chunk] for index,rank in enumerate(ranks)})
    def test_per_rank_and_node_totals(self):
        from TokenSim.comm.collectives import ring_traffic_factor
        for count in (1, 2, 4):
            for operation, expected in (("all_gather", (count-1)*100),
                                         ("reduce_scatter", (count-1)*100/count),
                                         ("all_reduce", 2*(count-1)*100/count)):
                self.assertEqual(ring_volume(operation, count, 100).send_bytes_per_rank, expected)
            self.assertEqual(ring_traffic_factor('all_gather', count)*count*100,
                             ring_volume('all_gather', count, 100).send_bytes_per_rank)
        nodes = {rank: 0 for rank in range(4)}
        plan = collective_plan('all_gather', tuple(nodes), 100, nodes)
        self.assertEqual(plan.wire_bytes, 1200)
        self.assertAlmostEqual(plan.service(CommunicationDomainConfig()), 1200/300e9)

    def test_hierarchical_gather_crosses_each_shard_once(self):
        nodes = {rank: rank//4 for rank in range(8)}
        plan = collective_plan('all_gather', tuple(nodes), 100, nodes)
        self.assertEqual(sum(phase.wire_bytes for phase in plan.phases if phase.phase_id.startswith('inter')), 800)
        self.assertEqual(plan.output_bytes_per_rank, 800)
        self.assertEqual([phase.logical_input_bytes for phase in plan.phases if phase.phase_id.startswith('local')], [200, 200])

    def test_reduction_phase_contract(self):
        nodes = {rank: rank//4 for rank in range(8)}
        plan = collective_plan('all_reduce', tuple(nodes), 128, nodes)
        self.assertEqual(plan.output_bytes_per_rank, 128)
        self.assertEqual({phase.operation for phase in plan.phases}, {'reduce_scatter', 'all_reduce', 'all_gather'})
        self.assertEqual([phase.logical_input_bytes for phase in plan.phases if phase.phase_id.startswith('inter')], [32]*4)
        scatter = collective_plan('reduce_scatter', tuple(nodes), 128, nodes)
        self.assertEqual(scatter.output_bytes_per_rank, 16)

    def test_invalid_inputs_and_partial_nodes(self):
        for ranks, size, nodes in (((), 16, {}), ((0,0), 16, {0:0}),
                                   ((0,1), 16, {0:0}), ((0,1,2), 16, {0:0,1:0,2:1})):
            with self.assertRaises(ConfigurationError):
                collective_plan('all_reduce', ranks, size, nodes)
        for size in (-1, 1.5, True, float('nan')):
            with self.assertRaises(ConfigurationError):
                ring_volume('all_reduce', 2, size)
        with self.assertRaises(ConfigurationError):
            ring_volume('reduce_scatter', 4, 100, 'bf16')
        singleton = collective_plan('all_reduce', (0,), 100, {0:0})
        self.assertEqual(singleton.service(CommunicationDomainConfig()), 0)

    def test_alpha_beta_ignores_legacy_efficiency(self):
        link = LinkClass('test', 'test', 1e9, 2, half_bandwidth_bytes=1e12)
        self.assertEqual(link.analytical_transfer_us(10000), 12)
        self.assertEqual(link.analytical_transfer_us(10000, oversubscription=2), 22)
        other = replace(link, half_bandwidth_bytes=1, collective_efficiency=.1)
        self.assertEqual(other.analytical_transfer_us(10000), 12)
        self.assertNotEqual(link.transfer_us(10000), other.transfer_us(10000))
        from tests.test_af_executor import af_test_hardware
        from tests.test_af_parallel_planning import _cluster
        from tests.test_af_operator_provider import _transfer_query
        from TokenSim.af import TopologyAFTransferProvider
        cluster=_cluster(a_tp=2,a_dp=1,f_tp=2,f_dp=1)
        cluster.af.transfer_topology_mode='flat_link'
        provider=TopologyAFTransferProvider(af_test_hardware(),cluster,cluster.workers(),None)
        gather=provider.estimate(_transfer_query('all_gather',payload_bytes=128,endpoints=('a-pool:0','a-pool:1')))
        self.assertEqual(gather.details['model_version'],'node-domain-fifo-v1')
        self.assertEqual(gather.details['output_bytes_per_rank'],256)
        self.assertEqual(gather.details['wire_bytes'],128)
        self.assertEqual(gather.details['rounds'],1)


class LocalityTest(unittest.TestCase):
    def setUp(self):
        self.nodes = {rank: rank//4 for rank in range(12)}

    def remote(self, plan):
        return sum(message.wire_bytes for phase in plan.phases for messages in phase.rounds
                   for message in messages if message.source_node != message.target_node)

    def test_uniform_no_third_phase_and_local_skip(self):
        nodes = {rank: rank//4 for rank in range(8)}
        plan = all_to_all_plan(tuple(nodes), 800, nodes)
        self.assertEqual(plan.provenance, 'uniform_fallback')
        self.assertEqual(self.remote(plan), 3200)
        self.assertFalse(any(phase.phase_id.startswith('destination') for phase in plan.phases))
        self.assertTrue(all(message.source != message.target for phase in plan.phases for messages in phase.rounds for message in messages))

    def test_explicit_redistribution_cost(self):
        plan = all_to_all_plan(tuple(self.nodes), 1200, self.nodes, destination_redistribution=True)
        self.assertTrue(any(phase.phase_id.startswith('destination') and phase.wire_bytes > 0 for phase in plan.phases))

    def test_fraction_and_hotspot_preserve_remote_bytes(self):
        uniform = all_to_all_plan(tuple(self.nodes), 1200, self.nodes, remote_fraction=.5)
        hotspot = all_to_all_plan(tuple(self.nodes), 1200, self.nodes, remote_fraction=.5, load_imbalance=4)
        self.assertAlmostEqual(self.remote(uniform), 7200)
        self.assertAlmostEqual(self.remote(hotspot), 7200)
        self.assertNotEqual(tuple(phase.wire_bytes for phase in uniform.phases), tuple(phase.wire_bytes for phase in hotspot.phases))
        local = all_to_all_plan(tuple(self.nodes), 1200, self.nodes, remote_fraction=.1)
        self.assertLess(self.remote(local), self.remote(uniform))
        self.assertIsNone(hotspot.output_bytes_per_rank)
        self.assertAlmostEqual(sum(value for _,value in hotspot.expected_output_bytes), 14400)
        self.assertGreater(max(value for _,value in hotspot.expected_output_bytes),1200)

    def test_invalid_statistics_fail(self):
        for kwargs in ({'remote_fraction':-1}, {'local_fraction':.8,'remote_fraction':.8},
                       {'load_imbalance':float('nan')}, {'destination_redistribution':1}):
            with self.assertRaises(ConfigurationError):
                all_to_all_plan(tuple(self.nodes), 1200, self.nodes, **kwargs)


class DomainFIFOTest(unittest.TestCase):
    def test_atomic_wait_does_not_hold_source_or_allow_bypass(self):
        env = simpy.Environment()
        scheduler = DomainScheduler(env, ('tx0','rx1','rx2','other'))
        times = {}
        def task(name, keys, duration, delay=0):
            yield env.timeout(delay)
            times[name] = yield from scheduler.run(keys, duration, (name,))
        env.process(task('busy', ('rx1',), 10))
        env.process(task('a', ('tx0','rx1'), 3, 1))
        env.process(task('b', ('tx0','rx2'), 2, 2))
        env.process(task('disjoint', ('other',), 2, 2))
        def observe():
            yield env.timeout(3)
            self.assertNotIn('tx0', scheduler.owners)
        env.process(observe())
        env.run()
        self.assertEqual(times['a'], (1,10,13))
        self.assertEqual(times['b'], (2,13,15))
        self.assertEqual(times['disjoint'], (2,2,4))

    def test_ties_use_logical_identity(self):
        for order in (('c','b','a'), ('a','b','c')):
            env = simpy.Environment()
            scheduler = DomainScheduler(env, ('local',))
            starts = {}
            def task(name):
                starts[name] = (yield from scheduler.run(('local',), 1, (name,)))[1]
            for name in order:
                env.process(task(name))
            env.run()
            self.assertEqual(starts, {'a':0,'b':1,'c':2})

    def test_cancellation_releases_all_resources(self):
        for waiting in (False, True):
            env = simpy.Environment()
            scheduler = DomainScheduler(env, ('tx','rx'))
            holder = scheduler.request(('rx',), ('holder',)) if waiting else None
            results = []
            def victim():
                try:
                    yield from scheduler.run(('tx','rx'), 10, ('victim',))
                except simpy.Interrupt:
                    pass
            process = env.process(victim())
            def cancel():
                yield env.timeout(1)
                process.interrupt()
                yield process
                if holder:
                    scheduler.release(holder)
                results.append((yield from scheduler.run(('tx','rx'), 1, ('next',))))
            env.process(cancel())
            env.run()
            self.assertEqual(results, [(1,1,2)])
            self.assertFalse(scheduler.owners)
            self.assertFalse(scheduler.waiters)

    def test_cached_static_services_keep_intermediate_queue(self):
        env = simpy.Environment()
        scheduler = DomainScheduler(env, ('local','remote'))
        times = []
        def busy():
            yield from scheduler.run(('remote',), 10, ('0busy',))
        def phases():
            times.append((yield from scheduler.run(('local',), 3, ('1local',))))
            times.append((yield from scheduler.run(('remote',), 8, ('2remote',))))
        env.process(busy())
        env.process(phases())
        env.run()
        self.assertEqual(times, [(0,0,3), (3,10,18)])

    def test_source_target_minimum_and_local_independence(self):
        config = CommunicationDomainConfig(inter_tx_bandwidth_bytes_per_s=50e9,
                                           inter_rx_bandwidth_bytes_per_s=25e9)
        self.assertEqual(transmission_service(config, 0, 1, 25000), 1e-6)
        self.assertEqual(transmission_service(config, 0, 0, 300000), 1e-6)
        self.assertEqual(CommunicationDomainConfig(inter_tx_bandwidth_bytes_per_s='400 Gb/s').inter_tx_bandwidth_bytes_per_s, 50e9)
        self.assertEqual(CommunicationDomainConfig(inter_tx_bandwidth_bytes_per_s='50 GB/s').inter_tx_bandwidth_bytes_per_s, 50e9)
        for value in (True, -1, float('inf'), '300'):
            with self.assertRaises((ConfigurationError, ValueError)):
                CommunicationDomainConfig(local_bandwidth_bytes_per_s=value)
        with self.assertRaises(ValueError):
            CommunicationDomainConfig(parallel_links=6)


class PlacementTest(unittest.TestCase):
    def test_automatic_and_override_placement_is_reproducible(self):
        from tests.test_af_executor import af_test_hardware
        from tests.test_af_parallel_planning import _cluster
        from TokenSim.hardware.placement import build_topology_placement
        cluster = _cluster(a_tp=2,a_dp=1,f_tp=6,f_dp=1)
        for strategy, nodes in (('mixed',1), ('dedicated',2)):
            cluster.af.communication_domains = CommunicationDomainConfig(
                node_count=nodes, placement_strategy=strategy,
                device_indices={'a-pool:0':1})
            first = build_topology_placement(cluster, cluster.workers(), af_test_hardware())
            second = build_topology_placement(cluster, cluster.workers(), af_test_hardware())
            self.assertEqual(first,second)
            self.assertEqual(first.device_index(0),1)
            self.assertEqual(len(set(first.device_index_by_worker.values())),8)
            if strategy == 'dedicated':
                self.assertTrue(all(first.device_index(rank)//8 == 1 for rank in range(2,8)))

    def test_policy_conflicts_and_inventory_shortage_fail(self):
        from tests.test_af_executor import af_test_hardware
        from tests.test_af_parallel_planning import _cluster
        from TokenSim.hardware.placement import build_topology_placement
        cluster = _cluster(a_tp=2,a_dp=1,f_tp=6,f_dp=1)
        for config in (CommunicationDomainConfig(node_count=1,placement_strategy='dedicated'),
                       CommunicationDomainConfig(node_count=1,placement_strategy='mixed',device_indices={'f-pool:0':0})):
            cluster.af.communication_domains = config
            with self.assertRaises(ConfigurationError):
                build_topology_placement(cluster,cluster.workers(),af_test_hardware())


class IntegratedDomainTest(unittest.TestCase):
    def test_compact_plan_keys_release_routing_graphs(self):
        import gc
        import weakref
        from TokenSim.af.latency import AFRoutingEdge
        from tests.test_af_operator_provider import _transfer_query
        _,_,provider = self.run_case()
        edge=AFRoutingEdge(0,'decode',0,0,0,0,0,1,.125,0,0)
        reference=weakref.ref(edge)
        query=replace(_transfer_query(payload_bytes=100,endpoints=('a-pool:0','a-pool:1')),routing_edges=(edge,))
        original=provider.communication_plan(query)
        changed=provider.communication_plan(replace(query,routing_edges=(replace(edge,expert_id=2),)))
        self.assertIsNot(original,changed)
        del query,edge
        gc.collect()
        self.assertIsNone(reference())

    def test_cached_phases_reacquire_resources_and_finish_at_eighteen(self):
        from types import SimpleNamespace
        from TokenSim.af.models import AFBlock
        from TokenSim.comm.traffic import TrafficPhase,TrafficPlan,Transmission
        for cache in (True,False):
            executor,_,provider = self.run_case(cache=cache)
            env = executor.env
            origin = env.now
            local = TrafficPhase('local','send_recv',(0,1),2,600000,600000,
                ((Transmission(0,1,600000,0,0),),))
            inter = TrafficPhase('inter','send_recv',(0,4),2,300000,300000,
                ((Transmission(0,4,300000,0,1),),),('local',))
            plan = TrafficPlan('send_recv',(0,1,4),600000,300000,(local,inter))
            provider.phase_estimate(local)
            provider.phase_estimate(inter)
            self.assertAlmostEqual(plan.service(provider.domain_config),11e-6)
            block = AFBlock(88,0,'prefill',(0,),1,1,(1,),0,'af-link')
            step = SimpleNamespace(node_id='cached',direction='collective',logical_stage_id='cached',
                source_domain='a-domain',target_domain='a-domain',layer_type=None,metadata={})
            env.process(executor.resources.domain_scheduler.run(inter.resource_ids,10e-6,(-1,'busy')))
            result = env.process(executor._run_domain_plan(block,0,step,'cached:wire',(),plan))
            env.run()
            self.assertAlmostEqual(result.value[2]-origin,18e-6)
            event = next(event for event in executor.trace.events_for(88) if event.details['phase_id']=='inter')
            self.assertAlmostEqual(event.queue_duration,7e-6)
            self.assertAlmostEqual(event.ready_at-origin,3e-6)

    def test_plan_eviction_and_queue_changes_keep_static_templates(self):
        from tests.test_af_operator_provider import _transfer_query
        executor,_,provider = self.run_case()
        provider.estimate_cache_capacity = 1
        provider._plan_cache.clear()
        query = _transfer_query(payload_bytes=100,endpoints=('a-pool:0','a-pool:1'))
        cold = provider.communication_plan(query)
        provider.communication_plan(replace(query,payload_bytes=200))
        rebuilt = provider.communication_plan(query)
        self.assertEqual(cold,rebuilt)
        self.assertIsNot(cold,rebuilt)
        self.assertIs(rebuilt,provider.communication_plan(query))
        self.assertFalse(any(hasattr(rebuilt,field) for field in ('ready_at','start_at','end_at')))

    def test_multiphase_cancellation_leaves_no_resources_or_successor(self):
        from types import SimpleNamespace
        from TokenSim.af.models import AFBlock
        for waiting in (False,True):
            executor,_,provider = self.run_case(trace=False)
            env = executor.env
            scheduler = executor.resources.domain_scheduler
            plan = collective_plan('all_reduce',tuple(range(8)),8000000,provider.node_by_worker)
            block = AFBlock(99,0,'prefill',(0,),1,1,(1,),0,'af-link')
            step = SimpleNamespace(node_id='cancel',direction='collective',logical_stage_id='cancel',
                source_domain='a-domain',target_domain='a-domain',layer_type=None,metadata={})
            holders = [scheduler.request(phase.resource_ids,(-1,phase.phase_id))
                       for phase in plan.phases[:2]] if waiting else []
            succeeded = []
            def victim():
                try:
                    yield from executor._run_domain_plan(block,0,step,'cancel:wire',(),plan)
                    succeeded.append(True)
                except simpy.Interrupt:
                    pass
            process = env.process(victim())
            def cancel():
                yield env.timeout(1e-6)
                process.interrupt('test cancellation')
                yield process
                for holder in holders:
                    scheduler.release(holder)
                yield from scheduler.run(('domain:node0:local',),1e-6,('following',))
            env.process(cancel())
            env.run()
            self.assertFalse(succeeded)
            self.assertFalse(scheduler.waiters)
            self.assertFalse(scheduler.owners)
            self.assertFalse(executor.active_stages)

    def test_internal_timetable_counts_serial_and_disjoint_work(self):
        from TokenSim.comm.traffic import TrafficPhase,Transmission
        config = CommunicationDomainConfig(local_bandwidth_bytes_per_s=1e6,
            inter_tx_bandwidth_bytes_per_s=1e6,inter_rx_bandwidth_bytes_per_s=1e6)
        phase = TrafficPhase('local','send_recv',(0,1,2),3,5,5,
            ((Transmission(0,1,5,0,0),Transmission(1,2,5,0,0)),))
        self.assertAlmostEqual(phase.service(config),10e-6)
        self.assertEqual(phase.resource_ids,('domain:node0:local',))
        disjoint = replace(phase,rounds=((Transmission(0,1,5,0,1),Transmission(2,3,5,2,3)),))
        self.assertAlmostEqual(disjoint.service(config),5e-6)

    def test_matching_measured_phase_and_ambiguous_scope(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from TokenSim.af import TopologyAFTransferProvider
        from tests.test_af_executor import af_test_hardware
        from tests.test_af_operator_provider import _transfer_query
        executor, _, base = self.run_case()
        query = replace(_transfer_query(payload_bytes=100,endpoints=('a-pool:0','a-pool:1')),dtype='bf16')
        parameters = tuple((key,getattr(base.domain_config,key)) for key in (
            'local_bandwidth_bytes_per_s','inter_tx_bandwidth_bytes_per_s','inter_rx_bandwidth_bytes_per_s',
            'local_latency_us','inter_latency_us'))
        record = dict(model_version='node-domain-fifo-v1',algorithm='direct',phase_kind='transfer',
                      operation='send_recv',dtype='bf16',participants=2,nodes=1,logical_input_bytes=100,
                      wire_bytes=100,wire_schedule=(((0,0,100),),),domain_parameters=parameters,
                      timing_scope='transport_service',latency_us=7,source_id='test:phase-measurement')
        hardware = af_test_hardware()
        for scope in ('transport_service','eager_interval','unknown'):
            package = SimpleNamespace(meta=SimpleNamespace(dataset_version='phase-fixture-v1',
                extra={'communication_phase_services':({**record,'timing_scope':scope},)}))
            with patch.object(hardware,'operator_package',return_value=package):
                provider = TopologyAFTransferProvider(hardware,executor.cluster_config,
                    list(executor.resources.worker_configs.values()),base.placement,fallback='table_only')
            if scope == 'transport_service':
                estimate = provider.estimate(query)
                self.assertEqual(estimate.service_duration,7e-6)
                self.assertEqual(estimate.match_type,'exact')
                self.assertEqual(estimate.dataset_version,'phase-fixture-v1')
                with self.assertRaises(ConfigurationError):
                    provider.estimate(replace(query,payload_bytes=200))
            else:
                with self.assertRaisesRegex(ConfigurationError,'timing scope'):
                    provider.estimate(query)

    def test_plan_keys_include_resource_mapping_and_routing_features(self):
        from tests.test_af_operator_provider import _transfer_query
        executor,_,provider = self.run_case()
        query = _transfer_query(payload_bytes=100,endpoints=('a-pool:0','a-pool:1'))
        original = provider.communication_plan(query)
        self.assertIs(original,provider.communication_plan(replace(query,iteration_id=99,node_id='new-call')))
        remote = provider.communication_plan(replace(query,physical_endpoints=('a-pool:0','f-pool:0')))
        self.assertNotEqual(original.phases[0].resource_ids,remote.phases[0].resource_ids)
        changed = provider.communication_plan(replace(query,topology={'route_policy':'other'}))
        self.assertIsNot(original,changed)
        with self.assertRaises(AttributeError):
            provider.domain_config.local_bandwidth_bytes_per_s = 1
        self.assertLessEqual(len(provider._plan_cache),provider.estimate_cache_capacity)

    def test_hierarchical_gather_runs_local_reorder_without_network_bytes(self):
        from types import SimpleNamespace
        from TokenSim.af.models import AFBlock
        from TokenSim.af import OperatorTableAFComputeProvider
        from tests.test_af_executor import af_test_hardware
        executor, _, provider = self.run_case()
        hardware = af_test_hardware()
        executor.compute_provider = OperatorTableAFComputeProvider(hardware, hardware.model('TestModel'), fallback='analytical_only')
        ranks = tuple(range(8))
        plan = collective_plan('all_gather', ranks, 100, provider.node_by_worker)
        block = AFBlock(77,0,'prefill',(0,),1,1,(1,),0,'af-link')
        step = SimpleNamespace(node_id='gather', direction='collective',logical_stage_id='gather',
                               source_domain='a-domain', target_domain='a-domain',layer_type=None,metadata={})
        process = executor.env.process(executor._run_domain_plan(block,-1,step,'gather:transport',(),plan))
        executor.env.run()
        events = executor.trace.events_for(77)
        reorder = [event for event in events if event.event_type == 'compute' and event.operation == 'data_reorder']
        self.assertEqual(len(reorder),8)
        self.assertTrue(all(event.service_duration > 0 and event.details['wire_bytes'] == 0 for event in reorder))
        executor.trace.validate_invariants(77)

    def run_case(self, *, cache=True, trace=True):
        from tests.test_af_executor import af_test_hardware
        from tests.test_af_parallel_planning import _cluster
        from tests.test_af_communication_timing import ZeroCompute
        from TokenSim.af import AFExecutor, TopologyAFTransferProvider
        from TokenSim.hardware.placement import build_topology_placement
        from types import SimpleNamespace
        cluster = _cluster(a_tp=8, a_dp=1, f_tp=8, f_dp=1)
        cluster.af.communication_domains = CommunicationDomainConfig(node_size=4, node_count=4,
                                                                     local_latency_us=1, inter_latency_us=2)
        cluster.af.service_estimate_cache = cache
        cluster.af.trace_enabled = trace
        workers = cluster.workers()
        hardware = af_test_hardware()
        placement = build_topology_placement(cluster, workers, hardware)
        provider = TopologyAFTransferProvider(hardware, cluster, workers, placement, fallback='analytical_only')
        env = simpy.Environment()
        executor = AFExecutor(env, cluster, workers, ZeroCompute(), provider, 'TestModel', 'bf16', 1, 16)
        request = SimpleNamespace(id=0, is_prefill=True, prefill_len=1, prefill_compute_len=1, context_len=1)
        iteration = executor.submit(0, [request], 'prefill')
        env.process(executor.run_iteration(iteration))
        env.run()
        executor.trace.validate_invariants(iteration.iteration_id)
        return executor, iteration, provider

    def test_collective_phases_and_trace_keep_dependencies(self):
        executor, iteration, provider = self.run_case()
        phases = [event for event in executor.trace.events if ':phase:' in (event.node_id or '')]
        self.assertTrue(any(event.operation == 'reduce_scatter' for event in phases))
        self.assertTrue(any(event.operation == 'all_gather' for event in phases))
        self.assertTrue(all(not resource.startswith('link:') for event in executor.trace.events
                            for resource in event.resource_ids))
        self.assertFalse(executor.resources.domain_scheduler.owners)
        self.assertFalse(executor.resources.domain_scheduler.waiters)

    def test_cache_and_trace_switches_preserve_completion(self):
        rows = [self.run_case(cache=cache, trace=trace) for cache, trace in ((True,True),(False,True),(True,False))]
        self.assertEqual(len({iteration.makespan for _,iteration,_ in rows}), 1)
        traces = [[(event.node_id,event.ready_at,event.start_at,event.end_at,event.resource_ids)
                   for event in executor.trace.events] for executor,_,_ in rows[:2]]
        self.assertEqual(traces[0], traces[1])


if __name__ == '__main__':
    unittest.main()
