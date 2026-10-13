"""Audit native fixed costs by actual invocation and preserve ablation evidence."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def audit(directory, profile, batch):
    events=json.loads((directory/f'tokensim-b{batch}-decode-events.json').read_text())
    result=json.loads((directory/f'tokensim-b{batch}.json').read_text())
    assert result['routing_replay']['kind']=='captured'
    twice={'compute_a_attention_us','compute_a_post_norm_us','compute_f_moe_block_us'}
    expected={name:2 if name in twice or name.startswith(('send_','receive_')) else 1
              for name in profile['costs']}
    for iteration in range(31):
        step=[e for e in events if e['iteration_id']==iteration]
        counts=Counter(e['details'].get('cost_field') for e in step)
        for name,entry in profile['costs'].items():
            assert counts[name]==expected[name],(iteration,name,counts[name])
            for e in step:
                if e['details'].get('cost_field')==name:
                    assert math.isclose(e['service_duration']*1e6,entry['fixed_us'],abs_tol=1e-7)
                    assert e['details']['calibrated']
        assert not counts['host_submit_us'] and not counts['host_receive_submit_us']
        assert not counts['compute_submit_us']
        by_field={e['details'].get('cost_field'):e for e in step}
        readback=next(e for e in step if e['operation']=='sampled_token_readback')
        prepared=next(e for e in step if e['operation']=='a_input_prepare')
        assert by_field['engine_step_schedule_us']['end_at']<=prepared['start_at']
        book=by_field['engine_runner_bookkeeping_us']
        output=by_field['engine_output_process_us']
        assert readback['end_at']<=book['start_at']<=book['end_at']<=output['start_at']
        assert book['node_id'] in output['predecessor_ids']
        for e in step:
            if e['details'].get('cost_field')=='compute_f_moe_block_us':
                assert ':router:compute_submit' in e['node_id']
    return dict(batch=batch,steps=31,mean_step_us=result['decode']['mean_step_us'],
                expected_calls_per_step=expected,all_profile_costs_materialized=True,
                critical_chain_us=result['decode']['mean_critical_chain_us'],
                native_call_order_enabled=result['fixed_host_assumptions'].get('native_call_order_enabled',False),
                validation_not_assessed_here=['CUDA API/device launch calibration',
                                              'independent native component validation'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--root-prefix',required=True);p.add_argument('--batches',type=int,nargs='+',default=[1,8])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    profile=json.loads(a.profile.read_text())
    result=[audit(Path(f'{a.root_prefix}{b}'),profile,b) for b in a.batches]
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
