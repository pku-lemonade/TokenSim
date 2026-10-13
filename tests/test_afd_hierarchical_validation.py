import copy
import hashlib
import json
import unittest

from scripts.report_afd_hierarchical_validation import validate_windows


class HierarchicalValidationTest(unittest.TestCase):
    def fixture(self):
        bundle = dict(schema_version=3, training_batches=[1,8], regression_batches=[4,2,3],
                      heldout_batches=[5], batch5_used_for_fit=False,
                      host_profile={'evidence':{'heldout_batches':[5]}},
                      operation_profile={'evidence':{'heldout_batches':[5]}})
        native = dict(frozen_bundle_sha256=hashlib.sha256(json.dumps(bundle,indent=2).encode()).hexdigest(),
            windows=[dict(batch=b,window=w,output_sha256='same',profiler_enabled=False,
                hooks_enabled=False,samples=1000,mean_us=3000,p50_us=2990,p95_us=3500)
                for b in (1,8,4,2,3,5) for w in range(3)])
        return bundle,native

    def test_all_windows_and_frozen_bundle_required(self):
        bundle,native = self.fixture()
        self.assertEqual(validate_windows(bundle,native),(1,8,4,2,3,5))
        invalid=[]
        for patch in ({'mean_us':float('nan')},{'profiler_enabled':True},
                      {'samples':10},{'output_sha256':'changed'},{'window':1}):
            variant=copy.deepcopy(native);variant['windows'][0].update(patch);invalid.append(variant)
        invalid.append({**native,'frozen_bundle_sha256':'changed'})
        invalid.append({**native,'windows':native['windows'][:-1]})
        for variant in invalid:
            with self.assertRaises(ValueError): validate_windows(bundle,variant)


if __name__ == '__main__':
    unittest.main()
