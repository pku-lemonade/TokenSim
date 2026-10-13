import unittest
from scripts.summarize_afd_preposted_kernels import paired_interval,extract


class PrepostedServiceTest(unittest.TestCase):
    def test_waiting_kernel_occupancy_is_not_transfer_service(self):
        for wait in (100,5100,20100):
            result=paired_interval((wait,wait+9),(0,wait+8))
            self.assertEqual(result['device_service_us'],9)
            self.assertEqual(result['receiver_data_ready_us'],8)
            self.assertEqual(result['receiver_peer_start_wait_us'],wait)

    def test_inconsistent_cross_rank_clock_is_rejected(self):
        with self.assertRaises(ValueError): paired_interval((100,109),(0,99))

    def test_kernel_identity_excludes_unlabelled_warmup_and_duplicate_gpu_ranges(self):
        events=[dict(cat='kernel',name='ncclSendRecv',ts=-100,dur=10)]
        for i in range(840):
            name=f'exchange:512:0:0:{i}'
            events += [dict(cat='user_annotation',name=name,ph='X',ts=i*100,dur=90),
                       dict(cat='gpu_user_annotation',name=name,ph='X',ts=i*100+10,dur=80),
                       dict(cat='kernel',name='ncclSendRecv',ts=i*100+20,dur=9)]
        document=dict(baseTimeNanoseconds=1000,traceEvents=events)
        result=extract(document,0)
        self.assertEqual(result['exchange:512:0:0:0'],(21,30))
        self.assertEqual(len(result),840)
        events.append(dict(cat='kernel',name='ncclSendRecv',ts=21,dur=8))
        with self.assertRaises(ValueError): extract(document,0)


if __name__=='__main__': unittest.main()
