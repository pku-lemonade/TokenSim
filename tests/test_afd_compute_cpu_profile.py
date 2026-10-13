import copy
import unittest
from scripts.fit_afd_compute_cpu import fit,native_groups
from scripts.fit_afd_bookkeeping import fit as fit_bookkeeping
from scripts.collect_afd_frozen_validation import validation_batches
from tests import test_af_host_profile as host_tests


class ComputeCpuProfileTest(unittest.TestCase):
    def test_v4_validation_reserves_batch7_after_complete_costs(self):
        bundle=dict(schema_version=4,training_batches=[1,8],regression_batches=[4],heldout_batches=[7],
            batch7_used_for_fit=False,device_service_profile={'scope':'device'},transport_profile={'scope':'transport'},
            host_profile={'evidence':{'heldout_batches':[7]}},operation_profile={'evidence':{'heldout_batches':[7]}})
        self.assertEqual(validation_batches(bundle),((1,8,4,7),(7,)))
        for patch in ({'batch7_used_for_fit':True},{'heldout_batches':[6]},{'transport_profile':None}):
            with self.assertRaises(ValueError):validation_batches({**bundle,**patch})

    def test_hierarchical_validation_reserves_unseen_batch(self):
        bundle = dict(schema_version=3, training_batches=[1,8], regression_batches=[4,2,3],
                      heldout_batches=[5], batch5_used_for_fit=False,
                      host_profile={'evidence': {'heldout_batches': [5]}},
                      operation_profile={'evidence': {'heldout_batches': [5]}})
        self.assertEqual(validation_batches(bundle), ((1,8,4,2,3,5),(5,)))
        for patch in ({'heldout_batches': [3]}, {'batch5_used_for_fit': True},
                      {'host_profile': {'evidence': {'heldout_batches': [2]}}}):
            with self.assertRaises(ValueError):
                validation_batches({**bundle, **patch})

    @classmethod
    def setUpClass(cls):
        host_tests.HostProfileTest.setUpClass()
        cls.fixture=host_tests.HostProfileTest()

    def base(self):
        doc=self.fixture.document(self.fixture.runtime())
        doc.update(schema_version=2)
        doc['evidence']['heldout_batches']=[2]
        doc['environment']['gpu_pair']='A,F'
        return doc

    def samples(self):
        return [dict(role=role,batch=b,window=w,
            timing_scope='host_service_excluding_cuda_api_union',prior_sync_outside_timer=True,
            cuda_runtime_and_driver_callbacks=True,profiler_enabled=False,gpu_uuid=role.upper(),
            empty_probe_us=[.1]*100,sources={name:dict(file='source.py',line=1,sha256='hash') for name in native_groups(role)},
            groups={name:[dict(service_us=5,wall_us=7,excluded_cuda_api_us=2,cuda_api_calls=1)]*200 for name in native_groups(role)})
            for role in ('a','f') for b in (1,8) for w in range(3)]

    def test_compute_exclusion_and_group_composition(self):
        docs=self.samples();base=self.base();original=copy.deepcopy(base)
        result,report=fit(docs,base)
        self.assertEqual(result['costs']['compute_a_attention_us']['fixed_us'],10)
        self.assertEqual(result['costs']['compute_f_moe_block_us']['fixed_us'],5)
        self.assertNotIn('compute_f_router_us',result['costs'])
        self.assertEqual(base,original)
        for mode in ('missing','duplicate','uncounted','wrong_sum','profiler','heldout','probe'):
            rows=copy.deepcopy(docs)
            first=next(iter(rows[0]['groups'].values()))[0]
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows.append(rows[0])
            elif mode=='uncounted':first['cuda_api_calls']=0
            elif mode=='wrong_sum':first['wall_us']=4
            elif mode=='profiler':rows[0]['profiler_enabled']=True
            elif mode=='heldout':rows[0]['batch']=2
            elif mode=='probe':rows[0]['empty_probe_us']=[10]*100
            with self.subTest(mode=mode),self.assertRaises(ValueError):fit(rows,base)

    def test_bookkeeping_readback_deducted_once(self):
        docs=[dict(operation='runner_bookkeeping_cpu',batch=b,window=w,timing_scope='host_service',
            external_wait_excluded=True,prior_work_synchronize_outside_timer=True,profiler_enabled=False,
            readback_excluded=True,async_scheduling=False,routing_d2h=False,logprobs=False,gpu_uuid='A',source_sha256='source',
            samples=[dict(context=129,service_us=5,wall_us=25,excluded_readback_us=20,readback_calls=1)]*200)
            for b in (1,8) for w in range(3)]
        base=self.base()
        self.assertEqual(fit_bookkeeping(docs,base)['costs']['engine_runner_bookkeeping_us']['fixed_us'],5)
        docs[0]['samples'][0]['excluded_readback_us']=0
        with self.assertRaises(ValueError):fit_bookkeeping(docs,base)

    def test_old_frozen_bundle_and_new_holdout_contract(self):
        self.assertEqual(validation_batches({'schema_version':1}),((1,8,4),(4,)))
        bundle=dict(schema_version=2,training_batches=[1,8],regression_batches=[4],heldout_batches=[2],
            host_profile=self.base(),operation_profile=self.base())
        self.assertEqual(validation_batches(bundle),((1,8,4,2),(2,)))
        bundle['host_profile']['evidence']['heldout_batches']=[4]
        with self.assertRaises(ValueError):validation_batches(bundle)


if __name__=='__main__':unittest.main()
