"""Compact, scope-safe decode acceptance report from independent native windows."""
import argparse
import json
import statistics
import math
from pathlib import Path


def summarize(manifest,simulated,profile_fingerprint=None):
    result=[]
    for batch in (1,8,4):
        windows=[]
        for case in manifest['cases']:
            if case['mode']!='timing' or case['batch']!=batch:continue
            path=Path(case['result'])
            if not path.exists():continue
            native=json.loads(path.read_text())
            if native.get('profiler_during_measurement') is not False:
                raise ValueError('profiled window cannot enter acceptance')
            if (native.get('concurrency')!=batch or native.get('isl')!=128 or native.get('osl')!=32
                    or native.get('router')!='F' or native.get('repeat_token_min_match')!=1):
                raise ValueError('native workload/output consistency mismatch')
            calls=json.loads((path.parent/'calls.json').read_text())
            if not calls or any(len(call['step_ms'])!=32 for call in calls):
                raise ValueError('incomplete native cohort')
            raw=[step for call in calls for step in call['step_ms'][1:]]
            if any(type(step) not in (int,float) or not math.isfinite(step) or step<=0 for step in raw):
                raise ValueError('native step duration must be finite and positive')
            values=[step*1000 for step in raw]
            windows.append(dict(window=case['window'],samples=len(values),mean_us=statistics.mean(values),
                p50_us=statistics.median(values),p95_us=sorted(values)[max(0,int(.95*len(values))-1)]))
        row=dict(batch=batch,windows=windows,status='pending',component_status='not_verified')
        if len(windows)==3 and len({w['window'] for w in windows})==3 and batch in simulated:
            real=statistics.mean(w['mean_us'] for w in windows)
            predicted=simulated[batch]['decode']['mean_step_us']
            if type(predicted) not in (int,float) or not math.isfinite(predicted) or real<=0 or predicted<=0:
                raise ValueError('step duration must be finite and positive')
            ratio=max(real,predicted)/min(real,predicted)
            row.update(native_mean_us=real,simulated_mean_us=predicted,ratio=ratio,
                       step_order_of_magnitude_pass=ratio<10,
                       status='step_only_pass' if ratio<10 else 'step_failed')
        result.append(row)
    return dict(scope='decode whole step only; components and causal waits need separate evidence',
        frozen_profile_sha256=profile_fingerprint,full_alignment_passed=False,results=result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--simulator-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    simulated={b:json.loads(path.read_text()) for b in (1,8,4)
               if (path:=args.simulator_dir/f'tokensim-b{b}.json').exists()}
    frozen=args.manifest.parent/'frozen-profile.json'
    fingerprint=json.loads(frozen.read_text())['sha256'] if frozen.exists() else None
    report=summarize(manifest,simulated,fingerprint)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)


if __name__=='__main__':main()
