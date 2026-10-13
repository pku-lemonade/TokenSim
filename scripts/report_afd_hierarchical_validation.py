"""Audit refrozen hierarchical replays without converting partial evidence to full success."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.audit_afd_fixed_cost_replay import audit
from scripts.collect_afd_frozen_validation import validation_batches


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_windows(bundle, native):
    batches, _ = validation_batches(bundle)
    expected = hashlib.sha256(json.dumps(bundle, indent=2).encode()).hexdigest()
    if native.get('frozen_bundle_sha256') != expected:
        raise ValueError('native frozen bundle mismatch')
    windows = native['windows']
    if (len(windows) != len(batches)*3 or
            {(r['batch'], r['window']) for r in windows} != {(b,w) for b in batches for w in range(3)}):
        raise ValueError('missing or duplicate independent timing window')
    for batch in batches:
        selected = [r for r in windows if r['batch'] == batch]
        if len({r['output_sha256'] for r in selected}) != 1:
            raise ValueError('native outputs differ across windows')
        for row in selected:
            if row['profiler_enabled'] or row['hooks_enabled'] or row['samples'] < 310:
                raise ValueError('unqualified native timing window')
            if any(isinstance(row[k], bool) or not isinstance(row[k], (int,float)) or
                   not math.isfinite(row[k]) or row[k] <= 0 for k in ('mean_us','p50_us','p95_us')):
                raise ValueError('invalid timing statistics')
    return batches


def check_replay(root, bundle, batch):
    directory = root / f'b{batch}'
    result = load(directory / f'tokensim-b{batch}.json')
    events = load(directory / f'tokensim-b{batch}-decode-events.json')
    if result['runtime_fingerprint'] != bundle['host_profile']['runtime_fingerprint']:
        raise ValueError('simulation runtime mismatch')
    if not result['fixed_host_assumptions'].get('native_call_order_enabled'):
        raise ValueError('native call ordering required')
    if result['communication_domains'] != bundle['communication_domains']:
        raise ValueError('simulation domain configuration mismatch')
    summary = audit(directory, bundle['host_profile'], batch)
    if not math.isfinite(summary['mean_step_us']) or summary['mean_step_us'] <= 0:
        raise ValueError('invalid simulated step duration')
    operations = bundle['operation_profile']['operations']
    for iteration in range(31):
        step = [e for e in events if e['iteration_id'] == iteration]
        actual = [e for e in step if e['event_type'] == 'operation_service']
        if Counter(e['operation'] for e in actual) != Counter({name:1 for name in operations}):
            raise ValueError('operation invocation mismatch')
        for event in actual:
            if (not event['details']['calibrated'] or not math.isclose(
                    event['service_duration']*1e6, operations[event['operation']]['fixed_us'], abs_tol=1e-7)):
                raise ValueError('operation cost mismatch')
        wires = [e for e in step if e['event_type'] in ('transfer','control_transfer')]
        if len(wires) != 6:
            raise ValueError('expected two control and four layer data messages')
        for event in wires:
            if set(event['resource_ids']) != {'domain:node0:local'}:
                raise ValueError('GPU2/3 message unexpectedly uses inter-node domain')
            if event['event_type'] == 'transfer' and event['payload_bytes'] != batch*256*2:
                raise ValueError('hidden payload mismatch')
    summary['operation_and_local_domain_counts_verified'] = True
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('bundle','native','replay-root','source-root','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    bundle = load(args.bundle)
    native = load(args.native/'timing-summary.json')
    batches = validate_windows(bundle,native)
    for name, digest in bundle['implementation_sha256'].items():
        if hashlib.sha256((args.source_root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen source changed: '+name)
    rows = []
    for batch in batches:
        replay = check_replay(args.replay_root,bundle,batch)
        windows = [w for w in native['windows'] if w['batch'] == batch]
        simulated = replay['mean_step_us']
        ratios = [max(w['mean_us'],simulated)/min(w['mean_us'],simulated) for w in windows]
        rows.append(dict(batch=batch, heldout=batch in bundle['heldout_batches'], simulation=replay,
            native_mean_us=statistics.mean(w['mean_us'] for w in windows),
            native_samples=sum(w['samples'] for w in windows), window_ratios=ratios,
            step_order_pass=all(r<10 for r in ratios)))
    report = dict(scope='refrozen hierarchical same-node step validation',
        frozen_hashes_verified=True, results=rows, all_steps_pass=all(r['step_order_pass'] for r in rows),
        full_alignment_passed=False, unmeasured=bundle['unmeasured'])
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
