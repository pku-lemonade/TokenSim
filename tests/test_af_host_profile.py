import copy
import unittest

from tests import test_af_native_protocol as native
from TokenSim.af.calibration import HostOverheadProfile, runtime_fingerprint, HOST_COVERAGE
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.errors import ConfigurationError


class HostProfileTest(unittest.TestCase):
    def test_v3_reserves_batch5_without_changing_old_schema(self):
        runtime = self.runtime()
        profile = self.document(runtime)
        profile['schema_version'] = 3
        profile['evidence']['heldout_batches'] = [5]
        self.load(profile, runtime)
        for version, heldout in ((3, [2]), (2, [5]), (1, [5])):
            profile['schema_version'] = version
            profile['evidence']['heldout_batches'] = heldout
            with self.assertRaisesRegex(ConfigurationError, 'held-out'):
                self.load(profile, runtime)

    @classmethod
    def setUpClass(cls):
        native.NativeProtocolTest.setUpClass()
        cls.fixture = native.NativeProtocolTest()

    def runtime(self, costs=None):
        doc = self.fixture.document(communication_timing=dict(enabled=True, native_control_enabled=True))
        if costs:
            doc['af']['communication_timing'].update(costs)
        runtime = build_model_runtime(doc, hardware=self.fixture.base.hardware, model=self.fixture.model)
        return runtime

    def test_domain_bandwidth_and_placement_invalidate_profile(self):
        original = self.runtime()
        doc = copy.deepcopy(original.doc)
        doc['af']['transfer_topology_mode'] = 'physical'
        doc['af']['communication_domains'] = dict(node_size=8, node_count=2,
            placement_strategy='manual', device_indices={'a-pool:0': 2, 'f-pool:0': 3})
        base = build_model_runtime(doc, hardware=original.hardware, model=original.model)
        profile = self.document(base)
        for changes in ({'local_bandwidth_bytes_per_s': 150e9},
                        {'local_latency_us': 2},
                        {'device_indices': {'a-pool:0': 2, 'f-pool:0': 8}}):
            altered = copy.deepcopy(doc)
            altered['af']['communication_domains'].update(changes)
            runtime = build_model_runtime(altered, hardware=original.hardware, model=original.model)
            with self.assertRaisesRegex(ConfigurationError, 'fingerprint mismatch'):
                self.load(profile, runtime)

    def document(self, runtime):
        return dict(schema_version=1, profile_id='unit-test-host', unit='us', timing_scope='host_service',
            contains_wait=False, excludes_device_launch=True, protocol='vllm_afd',
            runtime_fingerprint=runtime_fingerprint(runtime.plan, runtime.cluster, runtime.workers, runtime.hardware),
            environment=dict(execution_mode='eager', software_sha256='test-software'),
            batch_range=[1,8], context_range=[129,160],
            costs={name: dict(fixed_us=value, per_token_us=0, coverage=[HOST_COVERAGE[name]])
                   for name, value in [('a_step_prepare_us',10), ('host_submit_us',2)]},
            evidence=dict(source='unit-test://not-real-calibration', kind='synthetic_test',
                sample_count=100, window_count=3, training_batches=[1,8], heldout_batches=[4],
                method='constant', residual_us=0, validation_status='pending',
                quantiles_us={'a_step_prepare_us':dict(p50=10,p95=10),
                              'host_submit_us':dict(p50=2,p95=2)}))

    def load(self, doc, runtime=None, explicit=()):
        runtime = runtime or self.runtime()
        return HostOverheadProfile(doc, expected_runtime=runtime_fingerprint(
            runtime.plan, runtime.cluster, runtime.workers, runtime.hardware),
            expected_environment=dict(execution_mode='eager', software_sha256='test-software'),
            explicit_costs=explicit)

    def test_profile_overrides_defaults_without_adding_them(self):
        runtime = self.runtime()
        doc = copy.deepcopy(runtime.doc)
        profile = self.document(runtime)
        doc['af'].update(engine_overhead_profile=profile, calibration_environment=profile['environment'])
        runtime = build_model_runtime(doc, hardware=runtime.hardware, model=runtime.model)
        ex, rows, _, _ = simulate_decode(runtime, 1, 129, fallback='analytical_only')
        send = [e for e in ex.trace.events if e.details.get('cost_field') == 'host_submit_us']
        self.assertEqual(len(send), 6)
        self.assertTrue(all(abs(e.service_duration-2e-6)<1e-12 for e in send))
        self.assertTrue(all(e.source_id.startswith('profile:') for e in send))
        self.assertEqual(ex.overhead_profile.describe()['evidence_kind'], 'synthetic_test')
        baseline = self.runtime({'a_step_prepare_us':10, 'host_submit_us':2})
        _, reference, _, _ = simulate_decode(baseline, 1, 129, fallback='analytical_only')
        self.assertEqual(rows, reference)

    def test_explicit_default_value_is_still_a_conflict(self):
        runtime = self.runtime({'host_submit_us':5})
        self.assertIn('host_submit_us', runtime.cluster.af.communication_timing._explicit_cost_fields)
        with self.assertRaisesRegex(ConfigurationError, 'duplicate'):
            self.load(self.document(runtime), runtime, runtime.cluster.af.communication_timing._explicit_cost_fields)

    def test_measured_cost_origin_and_unique_control_charges(self):
        runtime = self.runtime()
        doc = copy.deepcopy(runtime.doc)
        profile = self.document(runtime)
        profile['evidence']['kind'] = 'measured'
        profile['costs'] = {name: dict(fixed_us=value, per_token_us=0,
                                      coverage=[HOST_COVERAGE[name]])
                            for name, value in [('control_encode_us', 7), ('control_decode_us', 11)]}
        profile['evidence']['quantiles_us'] = {name: dict(p50=entry['fixed_us'], p95=entry['fixed_us'])
                                              for name, entry in profile['costs'].items()}
        doc['af'].update(engine_overhead_profile=profile, calibration_environment=profile['environment'])
        runtime = build_model_runtime(doc, hardware=runtime.hardware, model=runtime.model)
        ex, _, _, _ = simulate_decode(runtime, 1, 129, steps=3, fallback='analytical_only')
        for name, expected in [('control_encode_us', 7), ('control_decode_us', 11)]:
            charged = [e for e in ex.trace.events if e.details.get('cost_field') == name]
            self.assertEqual(len(charged), 3)
            self.assertTrue(all(abs(e.service_duration*1e6-expected) < 1e-9 for e in charged))
            self.assertTrue(all(e.details['calibrated'] for e in charged))
            self.assertEqual(ex.host_cost_provenance(name, tokens=1)['origin'], 'measured_profile')
        missing = ex.host_cost_provenance('a_step_prepare_us', tokens=1)
        self.assertEqual(missing['origin'], 'unmeasured_default')
        self.assertFalse(missing['calibrated'])
        self.assertEqual(missing['value_us'], 0)

    def test_scope_coverage_environment_and_version_rejected(self):
        original = self.document(self.runtime())
        edits = [lambda d: d.update(contains_wait=True), lambda d: d.update(excludes_device_launch=False),
                 lambda d: d.update(timing_scope='inclusive'), lambda d: d.update(schema_version=2),
                 lambda d: d['environment'].update(software_sha256='changed'),
                 lambda d: d.update(runtime_fingerprint='changed'),
                 lambda d: d['costs']['host_submit_us'].update(coverage=['a_input_prepare']),
                 lambda d: d['costs']['host_submit_us'].update(fixed_us=-1),
                 lambda d: d['costs']['host_submit_us'].update(fixed_us=float('nan')),
                 lambda d: d['costs']['host_submit_us'].update(fixed_us=True),
                 lambda d: d['evidence'].update(window_count=1),
                 lambda d: d['evidence'].update(heldout_batches=[8])]
        edits += [lambda d: d.update(environment=5), lambda d: d.update(evidence=None),
                  lambda d: d['costs'].update(host_submit_us=5),
                  lambda d: d['costs']['host_submit_us'].update(coverage=5),
                  lambda d: d['costs']['host_submit_us'].update(per_token_us=1e308),
                  lambda d: d['evidence'].update(training_batches=5)]
        for edit in edits:
            doc=copy.deepcopy(original); edit(doc)
            with self.subTest(doc=doc), self.assertRaises(ConfigurationError):
                self.load(doc)

    def test_fingerprint_changes_with_content_and_workload_is_bounded(self):
        from types import SimpleNamespace
        doc = self.document(self.runtime())
        first = self.load(doc)
        doc['costs']['host_submit_us']['fixed_us']=3
        second=self.load(doc)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        for requests, phase in (([SimpleNamespace(context_len=128)],'decode'),
                                ([SimpleNamespace(context_len=129)]*9,'decode'),
                                ([SimpleNamespace(context_len=129)],'prefill')):
            with self.assertRaises(ConfigurationError): first.validate_workload(requests,phase)

    def test_standard_engine_keeps_model_boundaries_and_profile(self):
        import simpy
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.test_af_executor import _psla_config
        from TokenSim.config.config import ClusterConfig, KVTransferConfig
        from TokenSim.llm.llm_engine import LLMEngine
        from TokenSim.af.model_executor import ModelAFExecutor
        runtime = self.runtime()
        profile = self.document(runtime)
        doc = copy.deepcopy(runtime.doc)
        doc['af'].update(engine_overhead_profile=profile, calibration_environment=profile['environment'])
        env = simpy.Environment()
        with patch.object(runtime.hardware, 'model', return_value=runtime.model):
            engine = LLMEngine(env=env, block_size=16, batching='paged-attn',
                kv_transfer_config=KVTransferConfig.default(), psla_config=_psla_config(),
                cluster_config=ClusterConfig(**doc), hardware=runtime.hardware,
                prefill_worker_pool_type='round_robin', decode_worker_pool_type='round_robin',
                max_parallem_sum=8, latency_fallback='analytical_only')
        ex = engine.af_executor
        self.assertIsInstance(ex, ModelAFExecutor)
        request = SimpleNamespace(id=0, is_prefill=False, prefill_len=128,
                                  context_len=129, generation_idx=0)
        iteration = ex.submit(0, [request], 'decode')
        env.run(until=env.process(ex.run_iteration(iteration)))
        boundaries = {e.operation for e in ex.trace.events if e.layer_id == -1 and e.event_type == 'compute'}
        self.assertEqual(boundaries, {'embedding_lookup', 'final_norm', 'lm_head', 'token_selection'})
        self.assertEqual(ex.overhead_profile.describe()['profile_id'], 'unit-test-host')
        ex.trace.validate_invariants(iteration.iteration_id)

    def test_measured_compute_scope_and_combine_coverage(self):
        from types import SimpleNamespace
        from TokenSim.af.operator_provider import OperatorTableAFComputeProvider
        provider = OperatorTableAFComputeProvider(self.runtime().hardware, self.fixture.model,
            fallback='analytical_only', require_device_scope=True)
        estimate = SimpleNamespace(latency_us=1, match_type='exact', source_id='fixture', dataset_version='test')
        metadata = {}
        resolver = SimpleNamespace(query=lambda *args: estimate,
            package=SimpleNamespace(meta=SimpleNamespace(extra=metadata)), dataset_version='test')
        for scope in (None, 'inclusive', 'host_service', 'eager_interval'):
            metadata['timing_scopes'] = {'routed_experts':scope}
            with self.assertRaises(ConfigurationError):
                provider._component(resolver, 'experts', 'routed_experts', {})
        metadata['timing_scopes']['routed_experts'] = 'device_service'
        with self.assertRaises(ConfigurationError):
            provider._component(resolver, 'experts', 'routed_experts', {})
        metadata['operation_coverage'] = {'routed_experts':['routed_experts']}
        self.assertEqual(provider._component(resolver, 'experts', 'routed_experts', {}).latency_us, 1)
        metadata['operation_coverage']['routed_experts'] = ('routed_experts','weighted_combine')
        self.assertEqual(provider._component(resolver, 'experts', 'routed_experts', {}).covered_operations,
                         ('routed_experts','weighted_combine'))

    def test_same_link_name_with_changed_bandwidth_invalidates_profile(self):
        from dataclasses import replace
        from unittest.mock import patch
        runtime = self.runtime()
        original = self.document(runtime)
        lookup = runtime.hardware.link
        with patch.object(runtime.hardware, 'link', side_effect=lambda name: replace(
                lookup(name), bandwidth_per_direction_bytes_per_s=lookup(name).bandwidth_per_direction_bytes_per_s / 2)):
            with self.assertRaisesRegex(ConfigurationError, 'fingerprint'):
                self.load(original, runtime)

    def test_profile_content_separates_provider_namespaces(self):
        runtime = self.runtime()
        doc = copy.deepcopy(runtime.doc)
        profile = self.document(runtime)
        namespaces = []
        rows = []
        for cost in (2, 3):
            profile['costs']['host_submit_us']['fixed_us'] = cost
            doc['af'].update(engine_overhead_profile=profile, calibration_environment=profile['environment'])
            configured = build_model_runtime(doc, hardware=runtime.hardware, model=runtime.model)
            _, result, compute, transfer = simulate_decode(configured, 1, 129, fallback='analytical_only')
            namespaces.append((compute._estimate_namespace, transfer._estimate_namespace))
            rows.append(result)
        self.assertNotEqual(namespaces[0][0], namespaces[1][0])
        self.assertNotEqual(namespaces[0][1], namespaces[1][1])
        self.assertGreater(rows[1][0]['makespan_us'], rows[0][0]['makespan_us'])

    def test_native_boundary_failure_releases_resources_and_preserves_kv_owner(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from TokenSim.errors import SimulationStateError
        runtime = self.runtime()
        states = {state.physical_identity: state for state in runtime.plan.physical_states}
        self.assertGreater(states['a-pool:0'].kv_bytes_per_block, 0)
        self.assertEqual(states['f-pool:0'].kv_bytes_per_block, 0)
        ex, _, _, _ = simulate_decode(runtime, 1, 129, fallback='analytical_only')
        original = ex.compute_provider.estimate
        def fail(query):
            if query.operation == 'lm_head':
                raise ValueError('native boundary failure')
            return original(query)
        def iteration():
            request = SimpleNamespace(id=0, is_prefill=False, context_len=129, prefill_len=128, generation_idx=0)
            return ex.submit(0, [request], 'decode')
        with patch.object(ex.compute_provider, 'estimate', fail):
            with self.assertRaisesRegex(SimulationStateError, 'native boundary failure'):
                ex.env.run(until=ex.env.process(ex.run_iteration(iteration())))
        ex.env.run()
        for resource in (*ex.resources.compute_slots.values(),
                         *ex.resources.communication_resources.values(),
                         *ex.resources.host_submission_slots.values()):
            self.assertFalse(resource.users)
            self.assertFalse(resource.queue)
        resumed = iteration()
        ex.env.run(until=ex.env.process(ex.run_iteration(resumed)))
        ex.trace.validate_invariants(resumed.iteration_id)

    def test_native_buffers_are_reserved_once_and_reduce_kv_capacity(self):
        runtime=self.runtime()
        states={s.physical_identity:s for s in runtime.plan.physical_states}
        a,f=states['a-pool:0'],states['f-pool:0']
        self.assertEqual(dict(a.protocol_buffers)['a_return_hidden'],8*256*2)
        self.assertEqual(dict(f.protocol_buffers)['f_output_hidden'],8*256*2)
        self.assertEqual(len(dict(a.protocol_buffers)),len(a.protocol_buffers))
        raw=(a.capacity_bytes-a.model_bytes-a.protocol_buffer_bytes)//a.kv_bytes_per_block
        replica=next(r for r in runtime.plan.replicas if r.is_attention)
        self.assertEqual(replica.logical_gpu_blocks,raw-int(.01*raw))
        self.assertEqual(f.kv_bytes_per_block,0)
        from dataclasses import replace
        from unittest.mock import patch
        from TokenSim.hardware.device.sourced_value import SourcedValue
        original=runtime.hardware.device
        with patch.object(runtime.hardware,'device',side_effect=lambda name:replace(original(name),
                memory_capacity_bytes=SourcedValue(a.model_bytes+a.protocol_buffer_bytes-1,'test','D'),
                memory_reserved_bytes=SourcedValue(0,'test','D'))):
            with self.assertRaisesRegex(ConfigurationError,'required_bytes'):
                build_model_runtime(runtime.doc,hardware=runtime.hardware,model=runtime.model)

    def test_grouped_expert_submit_not_multiplied_by_topk(self):
        from dataclasses import replace
        for k in (1,2):
            runtime=self.runtime({'compute_submit_us':7})
            model=replace(runtime.model,moe=replace(runtime.model.moe,num_experts_per_tok=k))
            runtime=build_model_runtime(runtime.doc,hardware=runtime.hardware,model=model)
            ex,_,_,_=simulate_decode(runtime,8,129,fallback='analytical_only')
            experts=[e for e in ex.trace.events if e.event_type=='compute' and e.stage=='routed_expert']
            self.assertEqual(len(experts),2)
            for expert in experts:
                submits=[e for e in ex.trace.events if e.node_id==expert.node_id+':compute_submit']
                self.assertEqual(len(submits),1)
                self.assertAlmostEqual(submits[0].service_duration,7e-6)

    def test_fused_combine_keeps_dependency_without_second_service_or_submit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from TokenSim.af.latency import AFServiceEstimate
        runtime=self.runtime({'compute_submit_us':7})
        ex,_,_,_=simulate_decode(runtime,1,129,fallback='analytical_only')
        original=ex.compute_provider.estimate
        def fused(query):
            if query.stage=='routed_expert':
                return AFServiceEstimate(10e-6,'fused-oracle',details={
                    'covered_operations':('routed_experts','weighted_combine'),'timing_scope':'device_service'})
            if query.stage=='weighted_combine':
                raise AssertionError('combine must not be estimated again')
            return original(query)
        def submit():
            return ex.submit(0,[SimpleNamespace(id=0,is_prefill=False,context_len=129,prefill_len=128,generation_idx=0)],'decode')
        with patch.object(ex.compute_provider,'estimate',fused):
            iteration=submit()
            ex.env.run(until=ex.env.process(ex.run_iteration(iteration)))
        events=ex.trace.events_for(iteration.iteration_id)
        combined=[e for e in events if e.stage=='weighted_combine']
        self.assertEqual(len(combined),2)
        for event in combined:
            self.assertEqual(event.service_duration,0)
            self.assertTrue(event.details['semantic_completion_only'])
            self.assertFalse(any(e.node_id==event.node_id+':compute_submit' for e in events))
        self.assertFalse(ex._fused_combines)
        ex.trace.validate_invariants(iteration.iteration_id)
        resumed=submit()
        ex.env.run(until=ex.env.process(ex.run_iteration(resumed)))
        self.assertTrue(all(e.service_duration>0 for e in ex.trace.events_for(resumed.iteration_id)
                            if e.event_type=='compute' and e.stage=='weighted_combine'))


if __name__ == '__main__': unittest.main()
