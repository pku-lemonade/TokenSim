import copy
import unittest

from scripts.fit_afd_submit import fit
from tests import test_af_host_profile as host_tests


class SubmitProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host_tests.HostProfileTest.setUpClass()
        cls.fixture=host_tests.HostProfileTest()

    def documents(self):
        rows=[]
        for backend in ('torch','pynccl'):
            for batch in (1,8):
                for window in range(3):
                    for size,dtype in ([(8,'int64'),(179,'uint8')] if backend=='torch' else [(batch*512,'bfloat16')]):
                        for direction in (('a_to_f',) if backend=='torch' else ('a_to_f','f_to_a')):
                            for action in ('send','receive'):
                                for role,delay in [('none',0),('sender',5),('sender',20),('receiver',5),('receiver',20)]:
                                    rows.append(dict(backend=backend,batch=batch,window=window,bytes=size,dtype=dtype,
                                        direction=direction,action=action,delayed_role=role,delay_ms=delay,
                                        samples=[dict(service_us=7,returned_before_peer_entered=True)]*50))
        return [dict(rows=rows,sources={a:dict(file='source.py',sha256='hash') for a in ('send','receive')},
            gpu_uuids=['A','F'],timing_scope='cpu_api_entry_to_return',completion_sync_outside_timer=True,
            prior_device_work_synchronized=True,blocking_wait=False,profiler_enabled=False)]

    def test_delayed_peer_proof_and_full_grid_required(self):
        base=self.fixture.document(self.fixture.runtime())
        # Default synthetic fixture explicitly covers the generic sender cost.
        # New profile may replace defaults, but never another profile entry.
        base['costs'].pop('host_submit_us')
        base['evidence']['quantiles_us'].pop('host_submit_us')
        base['environment']['gpu_pair']='A,F'
        docs=self.documents()
        profile,report=fit(docs,base)
        self.assertEqual(len(report),6)
        self.assertTrue(all(r['fixed_us']==7 for r in report.values()))
        for mode in ('missing','peer_wait','nan','direction','duplicate'):
            invalid=copy.deepcopy(docs)
            if mode=='missing':invalid[0]['rows'].pop()
            elif mode=='duplicate':invalid[0]['rows'].append(invalid[0]['rows'][0])
            elif mode=='direction':invalid[0]['rows'][0]['direction']='f_to_a'
            elif mode=='nan':invalid[0]['rows'][0]['samples'][0]['service_us']=float('nan')
            else:
                row=next(r for r in invalid[0]['rows'] if r['action']=='send' and r['delayed_role']=='receiver')
                row['samples'][0]['returned_before_peer_entered']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):fit(invalid,base)


if __name__=='__main__':unittest.main()
