"""Check materialized preparation costs and native causal ordering per step."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def audit(directory,profile,batch):
    events=json.loads((directory/f'tokensim-b{batch}-decode-events.json').read_text())
    result=json.loads((directory/f'tokensim-b{batch}.json').read_text())
    iterations=sorted({e['iteration_id'] for e in events})
    assert len(iterations)==31 and result['routing_replay']['kind']=='captured'
    for iteration in iterations:
        step=[e for e in events if e['iteration_id']==iteration]
        operations=[e for e in step if e['event_type']=='operation_service']
        assert Counter(e['operation'] for e in operations)==Counter({name:1 for name in profile['operations']})
        by_name={e['operation']:e for e in operations}
        for name,event in by_name.items():
            assert math.isclose(event['service_duration']*1e6,profile['operations'][name]['fixed_us'],abs_tol=1e-7)
            assert event['details']['calibrated'] and event['resource_id'].startswith('host:')
        assert by_name['a_attention_metadata']['start_at']>=by_name['a_input_prepare']['end_at']
        for cost in ('control_encode_us','control_decode_us'):
            assert len([e for e in step if e['details'].get('cost_field')==cost])==1
        encoded=next(e for e in step if e['details'].get('cost_field')=='control_encode_us')
        decoded=next(e for e in step if e['details'].get('cost_field')=='control_decode_us')
        assert encoded['start_at']>=by_name['a_attention_metadata']['end_at']
        assert by_name['f_dp_metadata']['start_at']>=decoded['end_at']
        receive=next(e for e in step if e['layer_id']==0 and e['node_id'].endswith(':receive_submit'))
        assert receive['start_at']>=by_name['f_dp_metadata']['end_at']
        assert by_name['f_dp_metadata']['node_id'] in receive['predecessor_ids']
        assert not any(e['event_type']=='control_device' for e in step)
    return dict(batch=batch,steps=31,operation_calls=31*len(profile['operations']),
                mean_step_us=result['decode']['mean_step_us'],critical_chain_us=result['decode']['mean_critical_chain_us'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,8])
    a=p.parse_args()
    profile=json.loads(a.profile.read_text())
    result=[audit(Path(f'outputs/afd-preparation-replay-20261007-b{b}'),profile,b) for b in a.batches]
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
