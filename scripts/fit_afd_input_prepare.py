"""Extend the measured operation profile with audited input preparation."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from TokenSim.af.calibration import content_fingerprint
from TokenSim.af.operation_profile import OperationServiceProfile


def fit(documents, base):
    groups = {}
    sources = set()
    for document in documents:
        key = (document['batch'],document['window'])
        if key in groups or key not in {(b,w) for b in (1,8) for w in range(3)}:
            raise ValueError('invalid or duplicate input preparation window')
        if (document.get('operation') != 'a_input_prepare' or document.get('timing_scope') != 'operation_service'
            or document.get('completion') != 'device_synchronize_after_operation'
            or document.get('external_wait_excluded') is not True
            or document.get('prior_work_synchronize_outside_timer') is not True
            or document.get('profiler_enabled') is not False
            or document.get('gpu_uuid') != base['environment']['gpu_pair'].split(',')[0]):
            raise ValueError('unmatched preparation boundary/environment')
        samples = document['samples']
        if len(samples) != 200:
            raise ValueError('200 preparation samples required per window')
        for row in samples:
            if not 129 <= row['context'] <= 159:
                raise ValueError('preparation context outside calibrated decode scope')
            for name in ('service_us','api_us','completion_tail_us','excluded_prior_sync_us'):
                if type(row[name]) not in (int,float) or not math.isfinite(row[name]) or row[name] < 0:
                    raise ValueError('invalid preparation interval')
            if not math.isclose(row['service_us'],row['api_us']+row['completion_tail_us'],abs_tol=1e-8):
                raise ValueError('preparation interval accounting mismatch')
        groups[key] = [r['service_us'] for r in samples]
        sources.add(document['source_sha256'])
    if len(groups) != 6 or len(sources) != 1:
        raise ValueError('three windows and identical operation source required')
    result = copy.deepcopy(base)
    values = sorted(v for group in groups.values() for v in group)
    result['profile_id'] = 'pkua100-control-and-input-operations-20261007-v2'
    result['operations']['a_input_prepare'] = dict(
        fixed_us=statistics.mean(statistics.median(v) for v in groups.values()),
        coverage=['a_input_prepare'],input_ready='prior_device_work_complete',
        completion='device_synchronize_after_operation',resource_model='calling_thread_single_inflight')
    evidence = result['evidence']
    evidence['source'] += ';input-prepare:'+content_fingerprint(documents)
    evidence['statistics']['a_input_prepare'] = dict(p50_us=statistics.median(values),
        p95_us=values[math.ceil(len(values)*.95)-1],samples=len(values))
    evidence['audit'] += (' A input preparation starts after prior device synchronize outside timing; '
        'ends after own device synchronize, covering CPU organization, block/input copies, positions, '
        'seq lengths, slot mapping and logits indices only. No NCCL/model layer work. Explicit completion '
        'sync is included; measurement changes overlap and is not native step timing. Source SHA256='+next(iter(sources)))
    evidence['validation_status'] = 'pending'
    OperationServiceProfile(result,expected_runtime=base['runtime_fingerprint'],
                            expected_environment=base['environment'],explicit_costs=())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples',type=Path)
    parser.add_argument('--base-profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    documents = [json.loads(p.read_text()) for p in sorted(args.samples.glob('b*-w*.json'))]
    result = fit(documents,json.loads(args.base_profile.read_text()))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result['operations']['a_input_prepare']))


if __name__ == '__main__':
    main()
