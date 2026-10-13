"""Compare identical ready-input call boundaries with/without CUPTI; no fitting."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--sampled',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=[]
    for role in ('a','f'):
        for batch in (1,8):
            control=json.loads((a.control/f'{role}-b{batch}-w0.json').read_text())
            assert control['timing_scope']=='diagnostic_wall_not_fit'
            assert not control['cuda_runtime_and_driver_callbacks'] and control['prior_sync_outside_timer']
            windows=[json.loads((a.sampled/f'{role}-b{batch}-w{w}.json').read_text()) for w in range(3)]
            assert all(doc['sources']==control['sources'] for doc in windows)
            for name,rows in control['groups'].items():
                assert len(rows)==200
                assert all(row['cuda_api_calls']==0 for row in rows)
                wall=statistics.median(r['wall_us'] for r in rows)
                on=[statistics.median(r['wall_us'] for r in doc['groups'][name]) for doc in windows]
                cpu=[statistics.median(r['service_us'] for r in doc['groups'][name]) for doc in windows]
                results.append(dict(role=role,batch=batch,group=name,no_cupti_wall_median_us=wall,
                    cupti_wall_window_medians_us=on,cupti_cpu_window_medians_us=cpu,
                    cupti_wall_over_no_cupti=statistics.mean(on)/wall,
                    cupti_cpu_over_no_cupti_wall=statistics.mean(cpu)/wall))
    report=dict(scope='probe perturbation diagnostic only; not replacement host service',
        control_windows_per_batch_role=1,training_parameters_changed=False,
        limitations=['separate process windows include CPU frequency/scheduling variation',
                    'no-CUPTI wall includes CUDA API/own asynchronous submit, not pure CPU',
                    'both modes synchronize before calls; native overlap perturbation not isolated'],results=results)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(results))


if __name__=='__main__':main()
