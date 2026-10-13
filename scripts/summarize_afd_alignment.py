"""Summarize remote GPU traces without downloading their large event arrays."""
import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def summarize_trace(path, role, batch):
    events = json.loads(path.read_text())['traceEvents']
    steps = sorted([e for e in events if e.get('cat') == 'user_annotation'
                    and e.get('name', '').startswith('ProfilerStep#')], key=lambda e: e['ts'])
    starts = [s['ts'] for s in steps]
    external = {e.get('args', {}).get('External id'): e for e in events
                if e.get('cat') == 'cpu_op' and e.get('ph') == 'X'}
    by_step = defaultdict(lambda: defaultdict(float))
    decode_steps = set()
    prefill_steps = set()
    unmatched = 0
    for e in events:
        if role == 'attention' and e.get('cat') == 'cpu_op' and e.get('name') == 'aten::mm':
            dims = e.get('args', {}).get('Input Dims', [])
            if dims and dims[0] == [128 * batch, 256]:
                index = bisect.bisect_right(starts, e['ts']) - 1
                if index >= 0:
                    prefill_steps.add(index)
        if role == 'ffn' and e.get('cat') == 'cpu_op' and e.get('name', '').endswith('topk_softmax'):
            dims = e.get('args', {}).get('Input Dims', [])
            if dims and dims[0] and dims[0][0] == batch:
                index = bisect.bisect_right(starts, e['ts']) - 1
                if index >= 0:
                    decode_steps.add(index)
        if e.get('cat') != 'kernel':
            continue
        origin = external.get(e.get('args', {}).get('External id'))
        if origin is None:
            unmatched += 1
            continue
        index = bisect.bisect_right(starts, origin['ts']) - 1
        if index < 0 or origin['ts'] >= starts[index] + steps[index]['dur']:
            unmatched += 1
            continue
        name = e['name'].lower()
        if role == 'attention' and 'splitkv' in name:
            decode_steps.add(index)
        category = 'nccl_occupancy_including_peer_wait' if 'nccl' in name else 'gpu_compute'
        by_step[index][category] += e['dur']
        if role == 'ffn' and 'nccl' not in name:
            if 'topkgating' in name or 'gemv' in name or 'cutlass::kernel2' in name:
                by_step[index]['router_gpu'] += e['dur']
            else:
                by_step[index]['expert_combine_gpu'] += e['dur']
    decode_steps -= prefill_steps
    if not decode_steps:
        raise RuntimeError(f'No identified decode steps in {path}')
    labels = sorted({k for s in decode_steps for k in by_step[s]})
    return {'decode_steps': len(decode_steps), 'prefill_steps_excluded': len(prefill_steps),
            'unmatched_kernels': unmatched,
            'per_decode_median_us': {k: median([by_step[s][k] for s in decode_steps]) for k in labels},
            'per_decode_min_max_us': {k: [min(by_step[s][k] for s in decode_steps),
                                        max(by_step[s][k] for s in decode_steps)] for k in labels},
            'method': 'GPU kernel duration sums assigned via CPU External id and ProfilerStep; '
                      'decode selected by splitkv/topk shape; overlapping durations are not makespan; '
                      'NCCL occupancy includes waiting and is not isolated transfer service'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = []
    for batch in (1, 8):
        directory = args.root / f'c{batch}'
        calls = json.loads((directory / 'calls.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        row = {'batch': batch, 'unprofiled_decode_step_median_us':
               median([v * 1000 for c in calls for v in c['step_ms'][1:]]),
               'unprofiled_prefill_step_median_us': median([c['step_ms'][0] * 1000 for c in calls]),
               'unprofiled_call_median_us': median([c['elapsed_ms'] * 1000 for c in calls]),
               'repetitions': len(calls), 'repeat_token_min_match': result['repeat_token_min_match']}
        for role in ('attention', 'ffn'):
            row[role] = summarize_trace(next((directory / role).glob('*.pt.trace.json')), role, batch)
        rows.append(row)
    (args.root / 'native-summary.json').write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
