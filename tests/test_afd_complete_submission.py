import copy
import unittest
from scripts.fit_afd_complete_submission import qualify
from scripts.fit_afd_device_fixed_service import fit
from tests.test_af_host_profile import HostProfileTest
from TokenSim.errors import ConfigurationError


class CompleteSubmissionTest(unittest.TestCase):
    def test_api_unknown_wait_copy_and_source_drift_rejected(self):
        row=dict(service_us=4.,excluded_cuda_api_us=2.,wall_us=6.,cuda_api_calls=2)
        original=dict(timing_scope='host_service_excluding_cuda_api_union',prior_sync_outside_timer=True,
            role='a',batch=1,gpu_uuid='A',sources={'group':{'sha256':'source'}},groups={'group':[row]*200})
        audit=copy.deepcopy(original);audit['timing_scope']='diagnostic_api_audit_not_fit'
        for r in audit['groups']['group']:
            r['outer_cuda_apis']=[dict(name='cudaMemcpyAsync',duration_us=2.,copy_kinds=8)]
        qualify(original,audit)
        for mode in ('host_copy','sync','source','count'):
            d=copy.deepcopy(audit)
            if mode=='host_copy':d['groups']['group'][0]['outer_cuda_apis'][0]['copy_kinds']=2
            if mode=='sync':d['groups']['group'][0]['outer_cuda_apis'][0]['name']='cudaDeviceSynchronize'
            if mode=='source':d['sources']['group']['sha256']='other'
            if mode=='count':d['groups']['group'][0]['cuda_api_calls']=3
            with self.assertRaises(ValueError):qualify(original,d)

    def test_device_fit_uses_gpu_services_and_excludes_slot_mapping(self):
        rows=[];bases={}
        for b in (1,8):
            for floor in (0,1):bases[floor,b]={'a-pool:0':10+2*floor,'f-pool:0':5+floor}
            for w in range(3):
                rows.append(dict(batch=b,window=w,
                    attention=dict(unclassified={},decode_steps=31,mean_us={'total':26,'outside/kv_slot_mapping':10}),
                    ffn=dict(unclassified={},decode_steps=31,mean_us={'total':8})))
        value,_=fit(rows,bases);self.assertAlmostEqual(value,3)
        rows[0]['batch']=7
        with self.assertRaises(ValueError):fit(rows,bases)

    def test_submission_version_requires_audited_scope_and_new_holdout(self):
        HostProfileTest.setUpClass();fixture=HostProfileTest();runtime=fixture.runtime()
        d=fixture.document(runtime)
        d.update(schema_version=4,timing_scope='host_submission_service',excludes_device_launch=False,
            device_service_source_id='device_service:hash',submission_audit=dict(source='audit',
                source_hashes_verified=True,call_counts_verified=True,blocking_apis_absent=True,copy_kinds=[3]))
        d['evidence']['heldout_batches']=[7]
        fixture.load(d,runtime)
        from TokenSim.af.factory import build_model_runtime,simulate_decode
        doc=copy.deepcopy(runtime.doc)
        doc['af'].update(engine_overhead_profile=d,calibration_environment=d['environment'])
        conflicting=build_model_runtime(doc,hardware=runtime.hardware,model=runtime.model)
        with self.assertRaisesRegex(ConfigurationError,'device-only calibration'):
            simulate_decode(conflicting,1,129,steps=1,fallback='analytical_only')
        for key,value in [('excludes_device_launch',True),('device_service_source_id','assumption'),('timing_scope','host_service')]:
            with self.assertRaises(ConfigurationError):fixture.load({**d,key:value},runtime)
        d['evidence']['heldout_batches']=[5]
        with self.assertRaises(ConfigurationError):fixture.load(d,runtime)


if __name__=='__main__':unittest.main()
