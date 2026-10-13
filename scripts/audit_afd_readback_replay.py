"""Verify measured readback is charged once after sampling on captured routing."""
import argparse
import json
import math
from pathlib import Path

from scripts.audit_afd_preparation_replay import audit


def check(directory, baseline, profile, batch):
    report = audit(directory,profile,batch)
    events = json.loads((directory/f'tokensim-b{batch}-decode-events.json').read_text())
    cost = profile['operations']['sampled_token_readback']['fixed_us']
    old = json.loads((baseline/f'tokensim-b{batch}.json').read_text())['decode']['mean_step_us']
    assert math.isclose(report['mean_step_us']-old,cost,abs_tol=1e-7)
    outputs = []
    for iteration in sorted({e['iteration_id'] for e in events}):
        step = [e for e in events if e['iteration_id']==iteration]
        sample = next(e for e in step if e['event_type']=='compute' and e['operation']=='token_selection')
        readback = next(e for e in step if e['operation']=='sampled_token_readback')
        assert sample['node_id'] in readback['predecessor_ids']
        assert readback['start_at']>=sample['end_at']
        assert math.isclose(readback['end_at'],max(e['end_at'] for e in step),abs_tol=1e-12)
        if outputs:
            assert min(e['ready_at'] for e in step)>=outputs[-1]
        outputs.append(readback['end_at'])
    return {**report,'before_us':old,'readback_us':cost,'readback_calls':len(outputs)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    profile=json.loads(args.profile.read_text(encoding='utf-8'))
    result=[check(Path(f'outputs/afd-readback-replay-20261009-b{b}'),
        Path(f'outputs/afd-preparation-replay-20261007-b{b}'),profile,b) for b in (1,8)]
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
