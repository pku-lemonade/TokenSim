"""Native-call coverage must be charged by invocation, not logical expert count."""
import copy
import unittest

from tests import test_af_operation_profile as operation_tests
from TokenSim.af.calibration import EXTENDED_HOST_COVERAGE, COMPUTE_CALL_FIELDS, MESSAGE_CALL_FIELDS
from TokenSim.af.factory import simulate_decode
from TokenSim.errors import ConfigurationError


class NativeCallProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        operation_tests.OperationProfileTest.setUpClass()
        cls.fixture = operation_tests.OperationProfileTest()

    def profile(self, fields):
        f = self.fixture
        doc = f.host.document(f.host.runtime())
        doc['schema_version'] = 2
        doc['evidence']['heldout_batches'] = [2]
        doc['costs'] = {name: dict(fixed_us=value, per_token_us=0,
            coverage=[EXTENDED_HOST_COVERAGE[name]]) for name, value in fields.items()}
        doc['evidence']['quantiles_us'] = {name: dict(p50=value,p95=value) for name,value in fields.items()}
        return doc

    def run_profile(self, fields, steps=2, **options):
        runtime = self.fixture.configured(self.fixture.sampled_profile(),
            engine_overhead_profile=self.profile(fields), **options)
        return simulate_decode(runtime,1,129,steps=steps,fallback='analytical_only',
            trace=options.get('trace_enabled',True))

    def test_native_groups_and_message_backends_have_unique_charges(self):
        fields = {name: 7 for name in COMPUTE_CALL_FIELDS.values()}
        fields.update({name: 3 for name in MESSAGE_CALL_FIELDS.values()})
        ex, _, _, _ = self.run_profile(fields)
        counts = {}
        for e in ex.trace.events:
            field = e.details.get('cost_field')
            if field in fields:
                counts[field] = counts.get(field,0)+1
                self.assertAlmostEqual(e.service_duration*1e6,fields[field])
        # Two layers per decode, one grouped expert invocation per layer.
        for name in COMPUTE_CALL_FIELDS.values():
            expected = 2 if name in ('compute_a_embedding_us','compute_a_final_norm_us',
                                     'compute_a_lm_head_us','compute_a_token_selection_us') else 4
            self.assertEqual(counts.get(name,0),expected,name)
        # Two data directions, and two metadata messages (length/payload).
        for name in MESSAGE_CALL_FIELDS.values():
            self.assertEqual(counts.get(name,0),4,name)
        for iteration in (0,1):
            ex.trace.validate_invariants(iteration)

    def test_schedule_and_output_once_ordered_and_cache_invariant(self):
        fields = dict(engine_step_schedule_us=13,engine_output_process_us=17)
        _, before, _, _ = self.run_profile(dict(engine_step_schedule_us=0,engine_output_process_us=0))
        ex, rows, _, _ = self.run_profile(fields)
        for old,new in zip(before,rows):
            self.assertAlmostEqual(new['makespan_us']-old['makespan_us'],30)
        for iteration in (0,1):
            events = ex.trace.events_for(iteration)
            schedule = [e for e in events if e.details.get('cost_field')=='engine_step_schedule_us']
            output = [e for e in events if e.details.get('cost_field')=='engine_output_process_us']
            self.assertEqual(len(schedule),1)
            self.assertEqual(len(output),1)
            readback = next(e for e in events if e.operation=='sampled_token_readback')
            self.assertGreaterEqual(output[0].start_at,readback.end_at)
            self.assertIn(readback.node_id,output[0].predecessor_ids)
            self.assertEqual(output[0].end_at,max(e.end_at for e in events))
            ex.trace.validate_invariants(iteration)
        _, quiet, _, _ = self.run_profile(fields,trace_enabled=False,
            service_estimate_cache=False,dag_template_cache=False)
        self.assertEqual(rows,quiet)

    def test_generic_explicit_or_profile_overlap_rejected(self):
        for field,generic in [('compute_f_expert_group_us','compute_submit_us'),
                              ('send_pynccl_a_to_f_us','host_submit_us'),
                              ('receive_pynccl_f_to_a_us','host_receive_submit_us')]:
            profile = self.profile({field:7})
            with self.assertRaisesRegex(ConfigurationError,'duplicate'):
                self.fixture.host.load(profile,explicit=(generic,))
            old = self.fixture.host.document(self.fixture.host.runtime())
            if generic not in old['costs']:
                from TokenSim.af.calibration import HOST_COVERAGE
                old['costs'][generic] = dict(fixed_us=1,per_token_us=0,coverage=[HOST_COVERAGE[generic]])
            mixed = copy.deepcopy(profile)
            mixed['costs'][generic] = old['costs'][generic]
            mixed['evidence']['quantiles_us'][generic] = dict(p50=1,p95=1)
            with self.assertRaisesRegex(ConfigurationError,'duplicate'):
                self.fixture.host.load(mixed)

    def test_output_requires_cpu_token_ready_operation(self):
        runtime = self.fixture.configured(engine_overhead_profile=self.profile({'engine_output_process_us':7}))
        with self.assertRaisesRegex(ConfigurationError,'sampled_token_readback'):
            simulate_decode(runtime,1,129,fallback='analytical_only')

    def test_native_f_moe_call_includes_router_experts_and_combine_once(self):
        ex,_,_,_=self.run_profile({'compute_f_moe_block_us':23})
        charged=[e for e in ex.trace.events if e.details.get('cost_field')=='compute_f_moe_block_us']
        self.assertEqual(len(charged),4)
        self.assertTrue(all(':router:compute_submit' in e.node_id for e in charged))
        self.assertTrue(all(abs(e.service_duration*1e6-23)<1e-8 for e in charged))
        self.assertFalse(any(e.details.get('cost_scope')=='per_native_call_group'
            and ':router:compute_submit' not in e.node_id for e in ex.trace.events))
        for duplicate in ('compute_f_router_us','compute_f_expert_group_us'):
            with self.assertRaisesRegex(ConfigurationError,'duplicate'):
                self.fixture.host.load(self.profile({'compute_f_moe_block_us':23,duplicate:7}))

    def test_bookkeeping_after_readback_before_frontend_once(self):
        ex,rows,_,_=self.run_profile({'engine_runner_bookkeeping_us':19,'engine_output_process_us':17})
        _,base,_,_=self.run_profile({'engine_runner_bookkeeping_us':0,'engine_output_process_us':0})
        for before,after in zip(base,rows):self.assertAlmostEqual(after['makespan_us']-before['makespan_us'],36)
        for iteration in (0,1):
            events=ex.trace.events_for(iteration)
            readback=next(e for e in events if e.operation=='sampled_token_readback')
            book=[e for e in events if e.details.get('cost_field')=='engine_runner_bookkeeping_us']
            output=next(e for e in events if e.details.get('cost_field')=='engine_output_process_us')
            self.assertEqual(len(book),1)
            self.assertGreaterEqual(book[0].start_at,readback.end_at)
            self.assertIn(readback.node_id,book[0].predecessor_ids)
            self.assertGreaterEqual(output.start_at,book[0].end_at)
            self.assertIn(book[0].node_id,output.predecessor_ids)
            ex.trace.validate_invariants(iteration)

    def test_whole_moe_and_bookkeeping_preserve_cache_and_device_service(self):
        fields={'compute_f_moe_block_us':23,'compute_a_attention_us':17,
                'engine_runner_bookkeeping_us':19,'engine_output_process_us':7}
        ex,rows,_,_=self.run_profile(fields)
        _,quiet,_,_=self.run_profile(fields,trace_enabled=False,
            service_estimate_cache=False,dag_template_cache=False)
        self.assertEqual(rows,quiet)
        baseline,_,_,_=self.run_profile({name:0 for name in fields})
        def device_events(executor):
            return {(e.iteration_id,e.node_id):e.service_duration for e in executor.trace.events
                    if e.event_type=='compute'}
        before,after=device_events(baseline),device_events(ex)
        self.assertEqual(set(before),set(after))
        for key in before:self.assertAlmostEqual(before[key],after[key],places=12)
        for iteration in (0,1):ex.trace.validate_invariants(iteration)

    def test_readback_fitter_rejects_incomplete_or_contaminated_evidence(self):
        from scripts.fit_afd_sampled_readback import fit
        base = self.fixture.document(self.fixture.host.runtime())
        base['environment']['gpu_pair'] = 'A,F'
        rows = [dict(operation='sampled_token_readback',batch=b,window=w,
            timing_scope='operation_service',external_wait_excluded=True,
            prior_work_synchronize_outside_timer=True,profiler_enabled=False,
            input_ready='sampled_tokens_device_ready',completion='cpu_api_return_data_ready',
            gpu_uuid='A',source_sha256='source',
            samples=[dict(context=129,service_us=11,shape=[b,1],dtype='torch.int32')]*200)
            for b in (1,8) for w in range(3)]
        before = copy.deepcopy(base)
        result = fit(rows,base)
        self.assertEqual(result['operations']['sampled_token_readback']['fixed_us'],11)
        self.assertEqual(result['evidence']['heldout_batches'],[2])
        self.assertEqual(result['evidence']['validation_status'],'pending')
        self.assertEqual(base,before)
        for invalid in (rows[:3],rows+[rows[0]]):
            with self.assertRaises(ValueError): fit(invalid,base)
        for key,value in [('prior_work_synchronize_outside_timer',False),('profiler_enabled',True),
                          ('batch',2),('gpu_uuid','F'),('source_sha256','changed')]:
            invalid = copy.deepcopy(rows)
            invalid[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): fit(invalid,base)


if __name__ == '__main__':
    unittest.main()
