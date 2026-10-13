"""Frozen partial calibration acceptance, with component gaps kept explicit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from scripts.audit_afd_preparation_replay import audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    bundle=json.loads(Path('docs/plans/afd-frozen-preparation-bundle-20261007.json').read_text())
    profile=json.loads(Path('docs/plans/afd-preparation-profile-20261007.json').read_text())
    codec=json.loads(Path('docs/plans/afd-measured-codec-profile-20261007.json').read_text())
    assert bundle['operation_profile']==profile and bundle['host_profile']==codec
    assert not bundle['batch4_used_for_fit']
    native=json.loads(Path('outputs/afd-frozen-validation-20261007-v1/timing-summary.json').read_text())
    expected=hashlib.sha256(json.dumps(bundle,indent=2).encode()).hexdigest()
    assert expected==native['frozen_bundle_sha256']
    rows=[]
    for batch in (1,8,4):
        windows=[w for w in native['windows'] if w['batch']==batch]
        assert len(windows)==3 and {w['window'] for w in windows}=={0,1,2}
        assert len({w['output_sha256'] for w in windows})==1
        assert all(not w['profiler_enabled'] and not w['hooks_enabled'] and w['samples']>=310
                   and math.isfinite(w['mean_us']) and w['mean_us']>0 for w in windows)
        sim=audit(Path(f'outputs/afd-preparation-replay-20261007-b{batch}'),profile,batch)
        mean=statistics.mean(w['mean_us'] for w in windows)
        ratios=[max(w['mean_us'],sim['mean_step_us'])/min(w['mean_us'],sim['mean_step_us']) for w in windows]
        rows.append(dict(batch=batch,heldout=batch==4,native_mean_us=mean,simulated_mean_us=sim['mean_step_us'],
                         mean_ratio=max(mean,sim['mean_step_us'])/min(mean,sim['mean_step_us']),
                         window_ratios=ratios,step_order_of_magnitude_pass=all(r<10 for r in ratios),
                         native_windows=windows,simulation_audit=sim))
    result=dict(scope='frozen_partial_calibration_decode_step_only',frozen_bundle_sha256=expected,
        all_step_points_pass=all(r['step_order_of_magnitude_pass'] for r in rows),
        full_alignment_passed=False,component_status='requires separate evidence',
        unmeasured=bundle['unmeasured'],results=rows)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:r[k] for k in ('batch','native_mean_us','simulated_mean_us','mean_ratio','step_order_of_magnitude_pass')} for r in rows]))


if __name__=='__main__':main()
