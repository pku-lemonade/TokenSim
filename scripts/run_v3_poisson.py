"""Finite Poisson-arrival full V3 decode experiment, with FIFO dynamic batching."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simpy
from TokenSim.af import AFTraceRecorder, build_operator_table_af_providers
from scripts.deepseek_v3_runtime import configuration
from TokenSim.af.model_executor import ModelAFExecutor


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def arrival_times(qps, count, seed):
    rng = random.Random(seed)
    times, now = [], 0.0
    for _ in range(count):
        now += rng.expovariate(qps)
        times.append(now)
    return times


def schedule(env, times, max_batch, serve, request_rows, batch_rows, queue_rows):
    """One iteration at a time; arrivals keep flowing while GPU work executes."""
    queue = simpy.Store(env)

    def arrivals():
        for request_id, arrival in enumerate(times):
            yield env.timeout(arrival - env.now)
            yield queue.put(request_id)
            queue_rows.append(dict(time_s=env.now, pending=len(queue.items), event='arrival'))

    def worker():
        count = 0
        while count < len(times):
            first = yield queue.get()
            ids = [first]
            while queue.items and len(ids) < max_batch:
                ids.append((yield queue.get()))
            start = env.now
            queue_rows.append(dict(time_s=start, pending=len(queue.items), event='dispatch'))
            completions = yield env.process(serve(ids))
            batch_rows.append(dict(batch_size=len(ids), start_s=start, end_s=env.now))
            for request_id in ids:
                end = completions[request_id]
                request_rows.append(dict(request_id=request_id, arrival_s=times[request_id],
                    start_s=start, complete_s=end, queue_ms=(start-times[request_id])*1000,
                    service_ms=(end-start)*1000, latency_ms=(end-times[request_id])*1000))
            count += len(ids)
    env.process(arrivals())
    env.process(worker())


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered)*fraction)-1)]


def run_case(args, qps):
    out = args.output / f'qps-{qps:g}'
    out.mkdir(parents=True, exist_ok=True)
    config = configuration(full_model=True, a_tp=args.a_tp, f_tp=args.f_ep,
        af_batch_size=8, device_id='a100_sxm_80g', network='nvlink3',
        hypothetical_f_memory_gib=1536)
    dump(out/'cluster.json', config.doc)
    dump(out/'runtime-plan.json', config.plan.summary())
    dump(out/'hardware.json', dict(devices={w.hardware: config.hardware.device(w.hardware).to_dict()
        for w in config.workers}, link=config.hardware.link('nvlink3').__dict__))
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(hardware=config.hardware,
        model=config.model, cluster_config=config.cluster, worker_configs=config.workers,
        placement=config.placement, dtype='bf16', fallback='analytical_only')
    executor = ModelAFExecutor(env, config.cluster, config.workers, compute, transfer,
        config.model.model_id, 'bf16', 61, config.model.hidden_size,
        moe_config=config.model.moe, random_seed=args.seed, runtime_plan=config.plan,
        trace_recorder=AFTraceRecorder(True))
    times = arrival_times(qps, args.requests, args.seed)
    request_rows, batch_rows, queue_rows = [], [], []
    validation = dict(iterations=0, events=0, layers=61, status='passed')
    with gzip.open(out/'operators.trace.json.gz', 'wt', encoding='utf-8', compresslevel=1) as stream:
        stream.write('{"displayTimeUnit":"us","metadata":{"timing":"SIMULATED analytical_only; absolute simulation microseconds"},"traceEvents":[')
        first_event = True

        def serve(ids):
            nonlocal first_event
            requests = [SimpleNamespace(id=i, is_prefill=False, prefill_len=2048,
                context_len=2048, generation_idx=0) for i in ids]
            iteration = executor.submit(0, requests, 'decode')
            yield env.process(executor.run_iteration(iteration))
            events = executor.trace.events
            assert {e.layer_id for e in events if e.layer_id >= 0} == set(range(61))
            by_id = {e.node_id: e for e in events}
            for e in events:
                assert e.service_duration >= -1e-12 and e.queue_duration >= -1e-12
                for predecessor in e.predecessor_ids:
                    assert predecessor in by_id, predecessor
                    assert by_id[predecessor].end_at <= e.start_at + 1e-9
                row = dict(name=e.operation or e.stage or e.event_type, cat=e.event_type,
                    ph='X', pid=1, tid=e.resource_id, ts=e.start_at*1e6,
                    dur=e.service_duration*1e6, args=e.as_dict())
                if not first_event:
                    stream.write(',')
                stream.write(json.dumps(row))
                first_event = False
            completions = {i: iteration.block_completed_at[b.block_id]
                for b in iteration.blocks for i in b.request_ids}
            validation['iterations'] += 1
            validation['events'] += len(events)
            # The next iteration has no trace dependency on this drained batch.
            executor.trace.clear()
            return completions

        schedule(env, times, args.max_batch, serve, request_rows, batch_rows, queue_rows)
        env.run()
        stream.write(']}')
    assert len(request_rows) == args.requests
    assert len({r['request_id'] for r in request_rows}) == args.requests
    for r in request_rows:
        assert r['arrival_s'] <= r['start_s'] <= r['complete_s']
        assert abs(r['latency_ms']-r['queue_ms']-r['service_ms']) < 1e-6
    end = max(r['complete_s'] for r in request_rows)
    completed_in_window = sum(r['complete_s'] <= times[-1] for r in request_rows)
    row = dict(qps=qps, requests=args.requests, realized_arrival_qps=args.requests/times[-1],
        window_s=times[-1], drained_end_s=end, drain_s=end-times[-1],
        completed_in_arrival_window=completed_in_window,
        backlog_at_last_arrival=args.requests-completed_in_window,
        max_pending=max(r['pending'] for r in queue_rows),
        completed_qps_including_drain=args.requests/end,
        completed_qps_during_arrivals=completed_in_window/times[-1],
        mean_latency_ms=sum(r['latency_ms'] for r in request_rows)/args.requests,
        p95_latency_ms=percentile([r['latency_ms'] for r in request_rows], .95),
        mean_queue_ms=sum(r['queue_ms'] for r in request_rows)/args.requests,
        p95_queue_ms=percentile([r['queue_ms'] for r in request_rows], .95),
        mean_service_ms=sum(r['service_ms'] for r in request_rows)/args.requests,
        mean_batch=args.requests/len(batch_rows), max_batch=max(r['batch_size'] for r in batch_rows))
    dump(out/'requests.json', sorted(request_rows, key=lambda r:r['request_id']))
    dump(out/'batches.json', batch_rows)
    dump(out/'queue.json', queue_rows)
    dump(out/'validation.json', validation)
    dump(out/'summary.json', row)
    print(json.dumps(row), flush=True)
    return row


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    qps = [r['qps'] for r in rows]
    axes[0].plot(qps, [r['mean_latency_ms'] for r in rows], 'o-', label='Mean total latency')
    axes[0].plot(qps, [r['p95_latency_ms'] for r in rows], 's--', label='P95 total latency')
    axes[0].plot(qps, [r['mean_queue_ms'] for r in rows], '^:', label='Mean scheduler wait')
    axes[0].set(ylabel='Latency (ms/request)', title='Latency and scheduler queue')
    axes[1].plot(qps, [r['completed_qps_during_arrivals'] for r in rows], 'o-', label='During arrival window')
    axes[1].plot(qps, [r['completed_qps_including_drain'] for r in rows], 's--', label='Including drain')
    axes[1].set(ylabel='Completed decode requests/s (=tokens/s)', title='Delivered throughput')
    axes[2].plot(qps, [r['backlog_at_last_arrival'] for r in rows], 'o-', label='Unfinished at last arrival')
    axes[2].plot(qps, [r['max_pending'] for r in rows], 's--', label='Maximum pending queue')
    axes[2].set(ylabel='Requests', title='Backlog')
    for ax in axes:
        ax.set(xscale='log', xlabel='Offered arrival rate (requests/s)')
        ax.set_xticks(qps, [f'{x:g}' for x in qps]); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Full V3 BF16 | Poisson arrivals | A100 SXM / NVLink3 | analytical simulation')
    fig.text(.5, .01, '64 requests/rate, one seed; cached 2048 tokens, one output token; finite pilot, not steady-state serving.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(output/'poisson-qps.png'); fig.savefig(output/'poisson-qps.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qps', type=float, nargs='+', default=[1,4,8,16,32,64,128])
    parser.add_argument('--requests', type=int, default=64)
    parser.add_argument('--a-tp', type=int, default=2)
    parser.add_argument('--f-ep', type=int, default=32)
    parser.add_argument('--max-batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v3-poisson-2a32f-20260920')
    args = parser.parse_args()
    if min(*args.qps, args.requests, args.a_tp, args.f_ep, args.max_batch) <= 0:
        parser.error('rates, requests, batch and rank counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = dict(vars(args), output=str(args.output), model='DeepSeek-V3 671B BF16',
        layers=61, device='a100_sxm_80g', network='nvlink3', cached_tokens=2048,
        output_tokens=1, af_batch_size=8, power_w_per_card=250,
        capacity_gib_per_f=1536, timing='analytical_only',
        scheduling='FIFO dynamic batching, max 32, no deliberate batch wait; one iteration at a time',
        scope='Finite pilot from empty queue, drain all requests; excludes prefill and serving CPU overhead')
    exp_path = args.output/'experiment.json'
    if exp_path.exists() and json.loads(exp_path.read_text()) != experiment:
        raise ValueError('Output already contains a different experiment; use a fresh output directory')
    dump(exp_path, experiment)
    rows = []
    for qps in args.qps:
        summary = args.output/f'qps-{qps:g}'/'summary.json'
        rows.append(json.loads(summary.read_text()) if summary.exists() else run_case(args, qps))
        dump(args.output/'results.json', rows)
    with (args.output/'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(args.output, rows)


if __name__ == '__main__':
    main()
