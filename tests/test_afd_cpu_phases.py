import unittest
from scripts.summarize_afd_cpu_phases import partition


class CPUPhaseTest(unittest.TestCase):
    def test_nested_calls_partition_wall_without_double_counting(self):
        def event(name,a,b): return dict(name=name,start_ns=a*1000,end_ns=b*1000)
        result=partition(dict(start_ns=0,end_ns=100000,events=[
            event('execute_model',10,90),event('_model_forward',20,80),
            event('control_payload',20,30),event('torch_send',22,29),
            event('decoder_layer',30,45),event('decoder_layer',50,65),
            event('sample_tokens',90,95)]))
        self.assertEqual(sum(result.values()),100)
        self.assertEqual(result['control_payload'],10)
        self.assertEqual(result['decoder_layer_0'],15)
        self.assertEqual(result['decoder_layer_1'],15)
        self.assertEqual(result['_model_forward'],20)
        self.assertNotIn('torch_send',result)

    def test_outside_step_interval_is_rejected(self):
        with self.assertRaises(AssertionError):
            partition(dict(start_ns=10,end_ns=20,events=[dict(name='execute_model',start_ns=0,end_ns=20)]))


if __name__=='__main__': unittest.main()
