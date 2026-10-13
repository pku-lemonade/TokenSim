import copy
import unittest
from scripts.report_afd_component_validation import compare
from scripts.report_afd_api_audit import summarize


class ComponentValidationTest(unittest.TestCase):
    def test_independent_windows_do_not_fit_constants(self):
        windows={'sample':{w:[12.]*200 for w in range(3)}}
        self.assertTrue(compare({'sample':10},windows)[0]['order_pass'])
        self.assertEqual(compare({'sample':10},windows)[0]['frozen_us'],10)
        for bad in ({'sample':{0:[12.]*200}}, {'other':windows['sample']},
                    {'sample':{w:[float('nan')]*200 for w in range(3)}}):
            with self.assertRaises(ValueError):compare({'sample':10},bad)
        self.assertFalse(compare({'sample':1},windows)[0]['order_pass'])

    def test_api_audit_does_not_authorize_wait_or_launch_fitting(self):
        sample=dict(excluded_cuda_api_us=7,outer_cuda_apis=[
            dict(name='cudaLaunchKernel',duration_us=2,calls=1),
            dict(name='cudaStreamSynchronize',duration_us=5,calls=1)])
        doc=dict(role='a',batch=1,window=0,timing_scope='diagnostic_api_audit_not_fit',
                 prior_sync_outside_timer=True,groups={'attention':[sample]*200})
        row=summarize(doc)['groups']['attention']
        self.assertTrue(row['unknown_api_present'])
        self.assertFalse(row['admissible_for_fixed_cost'])
        bad=copy.deepcopy(doc);bad['groups']['attention'][0]['excluded_cuda_api_us']=8
        with self.assertRaises(ValueError):summarize(bad)


if __name__=='__main__':unittest.main()
