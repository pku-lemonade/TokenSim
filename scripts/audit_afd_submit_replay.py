"""Audit actual profile use and unique CPU charges on captured native routing."""
from collections import Counter
import json
import math
from pathlib import Path


def main():
    profile=json.loads(Path('docs/plans/afd-submit-profile-20261009.json').read_text(encoding='utf-8'))
    result=[]
    for batch in (1,8):
        directory=Path(f'outputs/afd-submit-replay-20261009-b{batch}')
        events=json.loads((directory/f'tokensim-b{batch}-decode-events.json').read_text())
        document=json.loads((directory/f'tokensim-b{batch}.json').read_text())
        assert document['routing_replay']['kind']=='captured'
        for iteration in range(31):
            step=[e for e in events if e['iteration_id']==iteration]
            counts=Counter(e['details'].get('cost_field') for e in step)
            for name,entry in profile['costs'].items():
                assert counts[name]==(2 if name.startswith(('send_','receive_')) else 1),(name,counts[name])
                for e in step:
                    if e['details'].get('cost_field')==name:
                        assert math.isclose(e['service_duration']*1e6,entry['fixed_us'],abs_tol=1e-7)
                        assert e['details']['calibrated']
            assert not counts['host_submit_us'] and not counts['host_receive_submit_us']
            by_field={e['details'].get('cost_field'):e for e in step}
            readback=next(e for e in step if e['operation']=='sampled_token_readback')
            prepared=next(e for e in step if e['operation']=='a_input_prepare')
            assert by_field['engine_step_schedule_us']['end_at']<=prepared['start_at']
            assert by_field['engine_output_process_us']['start_at']>=readback['end_at']
        result.append(dict(batch=batch,steps=31,mean_step_us=document['decode']['mean_step_us'],
            covered_costs=list(profile['costs']),all_profile_costs_materialized=True,
            generic_send_receive_assumptions_replaced=True,
            remaining=['compute submission','runner bookkeeping','precise native preposted readiness','new frozen validation']))
    with Path('docs/plans/afd-submit-replay-audit-20261009.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
