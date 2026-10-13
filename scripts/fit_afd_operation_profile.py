"""Fit separately audited ready-input control operations, preserving raw scope."""
import argparse
import json
import math
import statistics
from pathlib import Path
from TokenSim.af.calibration import content_fingerprint
from TokenSim.af.operation_profile import CONTROL_OPERATIONS as OPERATIONS, OperationServiceProfile


def fit(rows, runtime, environment):
    pair = environment.get('gpu_pair', '').split(',')
    if len(pair) != 2 or not all(pair) or pair[0] == pair[1]:
        raise ValueError('operation samples require the recorded A/F GPU pair')
    groups = {}
    for row in rows:
        key = (row['operation'], row['batch'], row['window'])
        if key in groups or row['operation'] not in OPERATIONS or row['batch'] not in (1, 8):
            raise ValueError('duplicate or unsupported operation measurement')
        values = row['wall_us']
        role = 'a' if row['operation'] == 'h2d' else 'f'
        if row.get('role') != role or row.get('gpu_uuid') != pair[0 if role == 'a' else 1]:
            raise ValueError('sample hardware/role does not match profile environment')
        if (row['timing_scope'] != 'ready_input_cpu_api_interval_including_own_device_work_and_runtime'
            or row['prior_gpu_work_synchronized'] is not True or row['profiler_enabled'] is not False
            or len(values) != row['samples'] or len(values) < 100
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError('inadmissible ready-input measurement')
        groups[key] = values
    if set(groups) != {(op, b, w) for op in OPERATIONS for b in (1, 8) for w in range(3)}:
        raise ValueError('three independent windows per batch and operation required')
    operations, stats = {}, {}
    for op, (_, coverage) in OPERATIONS.items():
        groups_for_op = [v for (name, _, _), v in groups.items() if name == op]
        values = sorted(v for group in groups_for_op for v in group)
        operations[op] = dict(fixed_us=statistics.mean(statistics.median(v) for v in groups_for_op),
            coverage=[coverage], input_ready='prior_device_work_complete',
            completion='cpu_api_return_data_ready', resource_model='calling_thread_single_inflight')
        stats[op] = dict(p50_us=statistics.median(values), p95_us=values[math.ceil(len(values)*.95)-1], samples=len(values))
    result = dict(schema_version=1, profile_id='pkua100-control-operations-20261007-v1',
        timing_scope='operation_service', unit='us', runtime_fingerprint=runtime, environment=environment,
        batch_range=[1, 8], context_range=[129, 159], payload_bytes=179, operations=operations,
        evidence=dict(source='afd-control-staging-20261007-v1:'+content_fingerprint(rows), kind='measured',
            external_wait_excluded=True, method='equal_window_medians', windows=3,
            training_batches=[1,8], heldout_batches=[4], validation_status='pending', statistics=stats,
            audit='Prior torch.cuda.synchronize is outside timing; native-equivalent pageable CPU to(device) '
                  'uses default non_blocking=False; length tensor construction copies CPU data; item() and '
                  'cpu().numpy().tobytes() return CPU-visible data. Includes own runtime/copy completion, '
                  'excludes encode/decode, NCCL and later receive allocation. Microbenchmark, not engine validation.'))
    OperationServiceProfile(result, expected_runtime=runtime, expected_environment=environment, explicit_costs=())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--environment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = [r for p in sorted(args.samples.glob('window-*.json')) for r in json.loads(p.read_text())]
    result = fit(rows, json.loads(args.runtime.read_text())['runtime_fingerprint'], json.loads(args.environment.read_text()))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
