import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from scripts.convert_afd_capture import convert
from scripts.fit_afd_host_profile import fit
from scripts import afd_capture_hooks as hooks


class CollectionPipelineTest(unittest.TestCase):
    def test_metadata_benchmark_uses_native_wire_types(self):
        from scripts.measure_ready_peer_nccl import message_specs
        self.assertEqual(message_specs('torch'),[(8,'int64'),(179,'uint8'),(181,'uint8')])
        self.assertTrue(all(dtype=='bfloat16' for _,dtype in message_specs('pynccl')))

    def test_report_requires_three_windows_and_rejects_nan(self):
        from scripts.report_afd_acceptance import summarize
        with tempfile.TemporaryDirectory() as directory:
            cases=[]
            for window in range(3):
                root=Path(directory)/str(window);root.mkdir()
                (root/'result.json').write_text(json.dumps(dict(concurrency=4,isl=128,osl=32,router='F',
                    repeat_token_min_match=1,profiler_during_measurement=False)))
                (root/'calls.json').write_text(json.dumps([dict(step_ms=[10]+[.1]*31)]))
                cases.append(dict(mode='timing',batch=4,window=window,result=str(root/'result.json')))
            manifest=dict(cases=cases)
            simulated={4:dict(decode=dict(mean_step_us=1000))}
            report=summarize(manifest,simulated)
            self.assertFalse(report['full_alignment_passed'])
            self.assertEqual(report['results'][-1]['status'],'step_failed')
            self.assertEqual(summarize(dict(cases=cases[:2]),simulated)['results'][-1]['status'],'pending')
            simulated[4]['decode']['mean_step_us']=float('nan')
            with self.assertRaises(ValueError):summarize(manifest,simulated)

    def test_heldout_requires_frozen_profile_and_selects_independent_windows(self):
        from scripts.run_afd_collection import selected_cases
        manifest=dict(cases=[dict(mode=mode,batch=b,window=w) for mode in ('timing','capture')
                            for b in (1,8,4) for w in range(3 if mode=='timing' else 1)])
        self.assertEqual(len(selected_cases(manifest,'train',None)),6)
        self.assertEqual(len(selected_cases(manifest,'capture-train',None)),2)
        for phase in ('heldout','capture-heldout'):
            with self.assertRaises(ValueError):selected_cases(manifest,phase,None)
        self.assertEqual(len(selected_cases(manifest,'heldout',Path('profile'))),3)

    def samples(self):
        return dict(schema_version=1,runtime_fingerprint='test-runtime',
            environment=dict(execution_mode='eager',software_sha256='test'),
            source='unit-test://not-measured',kind='synthetic_test', samples=[
                dict(cost_field='host_submit_us',batch=b,context=c,window_id=str(w),duration_us=2+3*b,
                    phase='decode',timing_scope='host_service',contains_wait=False,
                    excludes_device_launch=True,coverage=['send_submit'],profiler_enabled=False,source='oracle')
                for b in (1,2,8) for w in range(3) for c in (129,159)])

    def test_fit_identifiable_nonnegative_model_and_reject_contamination(self):
        profile=fit(self.samples(),profile_id='test',method='nonnegative_linear')
        self.assertAlmostEqual(profile['costs']['host_submit_us']['fixed_us'],2)
        self.assertAlmostEqual(profile['costs']['host_submit_us']['per_token_us'],3)
        self.assertEqual(profile['evidence']['kind'],'synthetic_test')
        self.assertEqual(profile['evidence']['validation_status'],'pending')
        edits=[dict(batch=4),dict(contains_wait=True),dict(timing_scope='inclusive'),
               dict(excludes_device_launch=False),dict(profiler_enabled=True),dict(duration_us=-1),
               dict(duration_us=float('nan')),dict(coverage=['step_total'])]
        for edit in edits:
            data=self.samples();data['samples'][0].update(edit)
            with self.subTest(edit=edit),self.assertRaises(ValueError):fit(data,profile_id='test')
        data=self.samples()
        data['samples']=[r for r in data['samples'] if r['window_id']!='2']
        with self.assertRaises(ValueError):fit(data,profile_id='test')

    def records(self):
        result=[]
        for step in range(2):
            names=['alice','bob'] if step==0 else ['bob','alice']
            result.append(dict(kind='row_identity',decode_step=step,
                rows=[dict(row=i,native_request_id=name,token_id=128+step) for i,name in enumerate(names)]))
            for layer in range(2):
                result.extend([dict(kind='routing',decode_step=step,layer_id=layer,
                    expert_ids=[[0,1],[2,3]],routing_weights=[[.2,.8],[.4,.6]]),
                    dict(kind='layer_buffers',decode_step=step,layer_id=layer,input_shape=[2,256],
                         output_shape=[2,256],input_bytes=1024,output_bytes=1024)])
        return result

    def convert(self,records):
        return convert(records,runtime='test-runtime',environment={'execution_mode':'eager'},
                       source='unit-test',batch=2,layers=2,steps=2)

    def test_capture_join_respects_actual_row_order_and_rejects_gaps(self):
        replay=self.convert(self.records())
        selected=[r for r in replay['assignments'] if r['decode_step']==1 and r['request_id']==1 and r['layer_id']==0]
        self.assertEqual([r['expert_id'] for r in selected],[0,1])
        self.assertEqual([r['routing_weight'] for r in selected],[.2,.8])
        for records in (self.records()[:-1],self.records()+[self.records()[0]]):
            with self.assertRaises(ValueError):self.convert(records)
        records=self.records();records[2]['output_bytes']=2048
        with self.assertRaises(ValueError):self.convert(records)

    def test_attention_hook_is_bounded_and_preserves_return(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'capture.enabled').touch()
            expected=object()
            class Runner:
                input_batch=SimpleNamespace(req_ids=['actual-id'],num_reqs=1,num_computed_tokens_cpu=[128])
                def _prepare_inputs(self,scheduler):return expected
            with patch.dict('os.environ',AFD_CAPTURE_DIR=directory,AFD_CAPTURE_BATCH='1',AFD_CAPTURE_STEPS='1'),patch.object(hooks,'install_codecs'):
                hooks.install_attention(Runner)
                runner=Runner()
                scheduler=SimpleNamespace(num_scheduled_tokens={'actual-id':1})
                self.assertIs(runner._prepare_inputs(scheduler),expected)
                self.assertIs(runner._prepare_inputs(scheduler),expected)
            rows=[json.loads(line) for path in root.glob('capture-*.jsonl') for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['rows'][0]['native_request_id'],'actual-id')

    def test_prepare_requires_new_path_and_preserves_source(self):
        from scripts.prepare_afd_capture import prepare,sha
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            source=base/'profiling/afd-dbo-fixed-20261007';source.mkdir(parents=True)
            plugin=base/'src/afd-plugin-dbo-cache-20261007'
            worker=plugin/'afd_plugin/v1/worker';worker.mkdir(parents=True)
            for name in ('attention_model_runner.py','ffn_model_runner.py'):
                (worker/name).write_text('# fixture source\n')
            names=['afd_plugin/v1/worker/'+name for name in ('attention_model_runner.py','ffn_model_runner.py')]
            audit=base/'audit.json';audit.write_text(json.dumps(dict(source_hashes={name:sha(plugin/name) for name in names})))
            (source/'attention.py').write_text("def main():\n    assert 'afd-plugin-dbo-cache-20261007' in actual_plugin.parts, actual_plugin\n    nv.nvmlInit()\n")
            (source/'run_curve.py').write_text("def run():\n    value=str(base/'src/afd-plugin-dbo-cache-20261007')\n    assert not any(uuid in apps for uuid in uuids), 'Selected GPUs have other processes; stop without interference'\n")
            hook=base/'hooks.py';hook.write_text('# fixture\n')
            result=prepare(base,base/'prepared',hook,audit)
            self.assertEqual(len(result['cases']),12)
            self.assertEqual(result['status'],'prepared_not_executed')
            self.assertEqual(result['source_hashes'],{name:sha(plugin/name) for name in names})
            with self.assertRaises(ValueError):prepare(base,base/'prepared',hook,audit)
            (worker/'ffn_model_runner.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'drift'):prepare(base,base/'other',hook,audit)


if __name__=='__main__':unittest.main()
