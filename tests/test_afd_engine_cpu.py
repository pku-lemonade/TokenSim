import copy
import unittest
from unittest.mock import patch

from scripts import afd_engine_cpu_hooks as hooks
from scripts.fit_afd_engine_cpu import fit, FUNCTIONS, LABELS
from tests import test_af_host_profile as host_tests


class EngineCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host_tests.HostProfileTest.setUpClass()
        cls.fixture=host_tests.HostProfileTest()

    def test_complete_audited_windows_only_and_base_preserved(self):
        base=self.fixture.document(self.fixture.runtime())
        rows=[dict(batch=b,window=w,timing_scope='host_service',contains_wait=False,
            profiler_enabled=False,execution_mode='eager',async_scheduling=False,detokenize=False,
            kv_transfer_enabled=False,gpu_uuids=['A','F'],coverage=FUNCTIONS,
            sources={name:dict(file='source.py',line=1,sha256='hash') for name in LABELS},
            samples=[dict(schedule=10,update_from_output=2,process_outputs=3)]*200)
            for b in (1,8) for w in range(3)]
        original=copy.deepcopy(base)
        profile,report=fit(rows,base,['A','F'])
        self.assertEqual(profile['costs']['engine_step_schedule_us']['fixed_us'],10)
        self.assertEqual(profile['costs']['engine_output_process_us']['fixed_us'],5)
        self.assertEqual(profile['evidence']['heldout_batches'],[2])
        self.assertEqual(base,original)
        for invalid in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError): fit(invalid,base,['A','F'])
        for key,value in [('contains_wait',True),('profiler_enabled',True),('batch',4),
                          ('gpu_uuids',['F','A']),('coverage',[])]:
            invalid=copy.deepcopy(rows);invalid[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): fit(invalid,base,['A','F'])
        invalid=copy.deepcopy(rows);invalid[0]['samples'][0]['schedule']=float('nan')
        with self.assertRaises(ValueError):fit(invalid,base,['A','F'])

    def test_missing_or_overlapping_calls_not_zero_cost(self):
        valid=[dict(name=name,start_ns=i*1000,end_ns=i*1000+500) for i,name in enumerate(LABELS)]
        with patch.object(hooks,'events',valid):
            self.assertEqual(hooks.finish_step(),{name:.5 for name in LABELS})
        with patch.object(hooks,'events',valid[:-1]),self.assertRaises(ValueError):hooks.finish_step()
        invalid=copy.deepcopy(valid);invalid[0]['end_ns']=2000
        with patch.object(hooks,'events',invalid),self.assertRaises(ValueError):hooks.finish_step()


if __name__=='__main__':unittest.main()
