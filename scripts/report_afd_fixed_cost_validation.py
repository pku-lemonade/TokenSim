"""Verify immutable candidate and no-hook windows; never infer component success."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from scripts.audit_afd_fixed_cost_replay import audit


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    bundle=load('docs/plans/afd-frozen-fixed-cost-bundle-20261009.json')
    expected=hashlib.sha256(json.dumps(bundle,indent=2).encode()).hexdigest()
    native=load(a.native/'timing-summary.json')
    assert native['frozen_bundle_sha256']==expected
    assert bundle['training_batches']==[1,8] and bundle['heldout_batches']==[2]
    assert not bundle['batch2_used_for_fit']
    assert bundle['host_profile']==load('docs/plans/afd-compute-cpu-profile-20261009.json')
    assert bundle['operation_profile']==load('docs/plans/afd-readback-profile-20261009.json')
    for path,digest in bundle['implementation_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,('frozen implementation changed',path)
    assert {(w['batch'],w['window']) for w in native['windows']}=={(b,w) for b in (1,8,4,2) for w in range(3)}
    assert len(native['windows'])==12
    results=[]
    for b in (1,8,4,2):
        windows=[w for w in native['windows'] if w['batch']==b]
        assert len({w['output_sha256'] for w in windows})==1
        assert all(not w['profiler_enabled'] and not w['hooks_enabled'] and w['samples']>=310
                   and math.isfinite(w['mean_us']) and w['mean_us']>0 for w in windows)
        sim=audit(Path(f'outputs/afd-compute-cpu-replay-20261009-b{b}'),bundle['host_profile'],b)
        mean=statistics.mean(w['mean_us'] for w in windows)
        ratios=[max(w['mean_us'],sim['mean_step_us'])/min(w['mean_us'],sim['mean_step_us']) for w in windows]
        results.append(dict(batch=b,heldout=b==2,regression=b==4,native_mean_us=mean,
            simulated_mean_us=sim['mean_step_us'],simulated_over_native=sim['mean_step_us']/mean,
            window_ratios=ratios,step_order_of_magnitude_pass=all(r<10 for r in ratios),
            windows=windows,simulation_audit=sim))
    result=dict(scope='frozen_candidate_partial_calibration_decode_step_only',
        frozen_bundle_sha256=expected,implementation_hashes_verified=True,
        all_step_points_pass=all(r['step_order_of_magnitude_pass'] for r in results),
        full_alignment_passed=False,unmeasured=bundle['unmeasured'],results=results)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:r[k] for k in ('batch','native_mean_us','simulated_mean_us','simulated_over_native')} for r in results]))


if __name__=='__main__':main()
