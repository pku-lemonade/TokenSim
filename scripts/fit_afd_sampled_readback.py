"""Append ready-input token readback with a new, pending validation generation."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from TokenSim.af.calibration import content_fingerprint
from TokenSim.af.operation_profile import OperationServiceProfile


def fit(documents, base):
    groups, sources = {}, set()
    for doc in documents:
        key = (doc['batch'],doc['window'])
        if (type(key[0]) is not int or type(key[1]) is not int or
                key not in {(b,w) for b in (1,8) for w in range(3)} or key in groups):
            raise ValueError('duplicate or non-training window')
        if (doc.get('operation') != 'sampled_token_readback'
                or doc.get('timing_scope') != 'operation_service'
                or doc.get('external_wait_excluded') is not True
                or doc.get('prior_work_synchronize_outside_timer') is not True
                or doc.get('profiler_enabled') is not False
                or doc.get('input_ready') != 'sampled_tokens_device_ready'
                or doc.get('completion') != 'cpu_api_return_data_ready'
                or doc.get('gpu_uuid') != base['environment']['gpu_pair'].split(',')[0]):
            raise ValueError('readback boundary/environment mismatch')
        source = doc.get('source_sha256')
        if not isinstance(source,str) or not source:
            raise ValueError('missing source hash')
        sources.add(source)
        rows = doc['samples']
        if len(rows) != 200:
            raise ValueError('200 samples per window required')
        for row in rows:
            value = row['service_us']
            if (type(value) not in (int,float) or not math.isfinite(value) or value < 0
                    or type(row['context']) is not int or not 129 <= row['context'] <= 159
                    or row.get('shape') != [key[0],1] or row.get('dtype') != 'torch.int32'):
                raise ValueError('invalid readback sample or token layout')
        groups[key] = [row['service_us'] for row in rows]
    if len(groups) != 6 or len(sources) != 1:
        raise ValueError('three independent windows per batch and stable source required')
    result = copy.deepcopy(base)
    result.update(schema_version=2,profile_id='pkua100-preparation-and-readback-20261008-v1')
    name = 'sampled_token_readback'
    values = sorted(v for group in groups.values() for v in group)
    result['operations'][name] = dict(
        fixed_us=statistics.mean(statistics.median(v) for v in groups.values()),
        coverage=[name],input_ready='sampled_tokens_device_ready',
        completion='cpu_api_return_data_ready',resource_model='calling_thread_single_inflight')
    evidence = result['evidence']
    evidence.update(heldout_batches=[2],validation_status='pending')
    evidence['statistics'][name] = dict(p50_us=statistics.median(values),
        p95_us=values[math.ceil(len(values)*.95)-1],samples=len(values))
    evidence['source'] += ';readback:'+content_fingerprint(documents)
    evidence['audit'] += (' 新验收代际：复用原profile '+base['profile_id']+'（内容指纹 '+
        content_fingerprint(base)+'），原batch4验收不改写、不作为新验收。新增batch2留出待测。'
        ' token已在设备就绪后计时原生_to_list至CPU数据可用；前序同步在计时外；源码 '+next(iter(sources)))
    OperationServiceProfile(result,expected_runtime=base['runtime_fingerprint'],
        expected_environment=base['environment'],explicit_costs=())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples',type=Path)
    parser.add_argument('--base-profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = fit([json.loads(p.read_text()) for p in sorted(args.samples.glob('b*-w*.json'))],
                 json.loads(args.base_profile.read_text(encoding='utf-8')))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)


if __name__ == '__main__':
    main()
