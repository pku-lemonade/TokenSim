"""Poisson requests from a fixed user population, with multi-token decode."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simpy
from TokenSim.af import AFTraceRecorder, build_operator_table_af_providers
from scripts.deepseek_v3_runtime import configuration
from TokenSim.af.model_executor import ModelAFExecutor
from scripts.run_v3_poisson import arrival_times, percentile


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def workload(qps, count, users, seed, unique_users=False):
    """Aggregate Poisson process with uniform or one-request-per-user marks."""
    rng = random.Random(seed + 1)
    if unique_users:
        if count > users:
            raise ValueError("unique user assignment requires requests <= users")
        marks = list(range(users))
        rng.shuffle(marks)
        marks = marks[:count]
    else:
        marks = [rng.randrange(users) for _ in range(count)]
    return [dict(request_id=i, user=marks[i], arrival_s=t)
            for i, t in enumerate(arrival_times(qps, count, seed))]


def schedule(env, arrivals, *, users, max_batch, output_tokens, context,
             serve, request_rows, token_rows, batch_rows, queue_rows,
             busy_policy='queue', progress=None):
    """One active request/user; per-user FIFO, round-robin decode tokens.

    QPS is independent of completions. All queued requests drain after the final
    arrival. A new request may arrive during an iteration; admission is at the
    next iteration boundary. Users with no arrived requests remain idle.
    """
    pending = [deque() for _ in range(users)]
    busy = [False] * users
    runnable = deque()
    wake = env.event()
    arrivals_done = False

    def snapshot(event):
        queue_rows.append(dict(time_s=env.now, event=event,
            pending=sum(map(len, pending)), active_users=sum(busy)))

    def admit(row, ready):
        busy[row['user']] = True
        row['eligible_s'] = ready
        runnable.append(SimpleNamespace(id=row['request_id'], slot=row['user'],
            is_prefill=False, prefill_len=context, context_len=context,
            generation_idx=0, ready=ready, last_token=None, row=row))

    def produce():
        nonlocal arrivals_done, wake
        for source in arrivals:
            yield env.timeout(source['arrival_s'] - env.now)
            row = dict(source, status='accepted')
            request_rows.append(row)
            user = row['user']
            if busy[user]:
                if busy_policy == 'drop':
                    row['status'] = 'dropped'
                else:
                    pending[user].append(row)
            else:
                admit(row, env.now)
            snapshot('arrival')
            if not wake.triggered:
                wake.succeed()
        arrivals_done = True
        if not wake.triggered:
            wake.succeed()

    def consume():
        nonlocal wake
        iteration = 0
        while True:
            if not runnable:
                if arrivals_done:
                    break
                wake = env.event()
                yield wake
                continue
            requests = [runnable.popleft() for _ in range(min(max_batch, len(runnable)))]
            assert len({r.slot for r in requests}) == len(requests)
            start = env.now
            for req in requests:
                req.row.setdefault('first_admitted_s', start)
            snapshot('dispatch')
            completions = yield env.process(serve(requests, iteration))
            batch_rows.append(dict(iteration=iteration, start_s=start, end_s=env.now,
                batch_size=len(requests), users=[r.slot for r in requests],
                request_ids=[r.id for r in requests]))
            for req in requests:
                end = completions[req.id]
                assert req.ready <= start + 1e-9 and start <= end <= env.now + 1e-9
                token_rows.append(dict(user=req.slot, request_id=req.id,
                    token_index=req.generation_idx, context_len=req.context_len,
                    ready_s=req.ready, admitted_s=start, completion_s=end,
                    scheduler_wait_ms=(start-req.ready)*1000,
                    service_ms=(end-start)*1000,
                    itl_ms=None if req.last_token is None else (end-req.last_token)*1000))
                req.row.setdefault('first_token_s', end)
                req.generation_idx += 1
                if req.generation_idx == output_tokens:
                    req.row.update(status='complete', completion_s=end,
                        output_tokens=output_tokens)
                    if pending[req.slot]:
                        row = pending[req.slot].popleft()
                        admit(row, max(end, row['arrival_s']))
                    else:
                        busy[req.slot] = False
                else:
                    req.context_len = context + req.generation_idx
                    req.ready = req.last_token = end
                    runnable.append(req)
            snapshot('iteration_complete')
            iteration += 1
            if progress is not None and (iteration == 1 or iteration % 100 == 0):
                progress(iteration, sum(busy), sum(map(len, pending)))
        assert not any(pending) and not any(busy)

    env.process(produce())
    env.process(consume())


def validate_and_summarize(args, qps, arrivals, requests, tokens, batches, queues):
    complete = [r for r in requests if r['status'] == 'complete']
    dropped = [r for r in requests if r['status'] == 'dropped']
    assert len(requests) == len(arrivals) == len(complete) + len(dropped)
    counts = Counter(t['request_id'] for t in tokens)
    assert len(tokens) == len(complete) * args.output_tokens
    by_request = {r['request_id']:r for r in complete}
    for r in complete:
        assert counts[r['request_id']] == args.output_tokens
        assert r['arrival_s'] <= r['eligible_s'] <= r['first_admitted_s'] <= r['first_token_s'] <= r['completion_s']
    previous = {}
    for token in tokens:
        r = by_request[token['request_id']]
        assert token['context_len'] == args.context + token['token_index']
        assert r['arrival_s'] <= token['ready_s'] <= token['admitted_s'] <= token['completion_s']
        key = token['request_id']
        assert token['token_index'] == previous.get(key, (-1, 0))[0] + 1
        assert token['ready_s'] >= previous.get(key, (-1, 0))[1] - 1e-9
        previous[key] = (token['token_index'], token['completion_s'])
    for user in range(args.users):
        rows = sorted((r for r in complete if r['user']==user), key=lambda r:r['arrival_s'])
        for left, right in zip(rows, rows[1:]):
            assert right['first_admitted_s'] >= left['completion_s'] - 1e-9
    assert all(b['batch_size'] <= args.max_batch and len(set(b['users'])) == b['batch_size'] for b in batches)
    assert max(q['active_users'] for q in queues) <= args.users
    end = max(r['completion_s'] for r in complete)
    last_arrival = arrivals[-1]['arrival_s']
    arrival_tokens = sum(t['completion_s'] <= last_arrival for t in tokens)
    last_complete = sum(r['completion_s'] <= last_arrival for r in complete)
    latencies = [(r['completion_s']-r['arrival_s'])*1000 for r in complete]
    ttfts = [(r['first_token_s']-r['arrival_s'])*1000 for r in complete]
    service_ttfts = [(r['first_token_s']-r['first_admitted_s'])*1000 for r in complete]
    return dict(qps=qps, users=args.users, observed_users=len({r['user'] for r in requests}),
        a_tp=args.a_tp, f_ep=args.f_ep, max_batch=args.max_batch, af_batch_size=args.af_batch_size,
        requests=len(requests), completed_requests=len(complete), dropped_requests=len(dropped),
        output_tokens_per_request=args.output_tokens, total_output_tokens=len(tokens),
        realized_arrival_qps=len(arrivals)/last_arrival,
        offered_token_rate=qps*args.output_tokens,
        arrival_window_s=last_arrival, drained_end_s=end, drain_s=end-last_arrival,
        total_tps=len(tokens)/end, tps_during_arrivals=arrival_tokens/last_arrival,
        completed_qps=len(complete)/end, mean_user_tps=len(tokens)/end/args.users,
        unfinished_at_last_arrival=len(complete)-last_complete,
        max_pending=max(q['pending'] for q in queues),
        max_active_users=max(q['active_users'] for q in queues),
        mean_batch=len(tokens)/len(batches), max_actual_batch=max(b['batch_size'] for b in batches),
        mean_latency_ms=sum(latencies)/len(latencies), p95_latency_ms=percentile(latencies,.95),
        mean_ttft_ms=sum(ttfts)/len(ttfts), p95_ttft_ms=percentile(ttfts,.95),
        mean_service_ttft_ms=sum(service_ttfts)/len(service_ttfts),
        p95_service_ttft_ms=percentile(service_ttfts,.95))


def run_case(args, qps):
    out = args.output/f'qps-{qps:g}'
    out.mkdir(parents=True, exist_ok=True)
    config = configuration(full_model=True, a_tp=args.a_tp, f_tp=args.f_ep,
        af_batch_size=args.af_batch_size, device_id='a100_sxm_80g', network='nvlink3',
        hypothetical_f_memory_gib=1536)
    dump(out/'cluster.json', config.doc)
    dump(out/'runtime-plan.json', config.plan.summary())
    dump(out/'hardware.json', dict(devices={w.hardware:config.hardware.device(w.hardware).to_dict()
        for w in config.workers}, link=config.hardware.link('nvlink3').__dict__))
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(hardware=config.hardware,
        model=config.model, cluster_config=config.cluster, worker_configs=config.workers,
        placement=config.placement, dtype='bf16', fallback='analytical_only')
    executor = ModelAFExecutor(env, config.cluster, config.workers, compute, transfer,
        config.model.model_id, 'bf16', 61, config.model.hidden_size,
        moe_config=config.model.moe, random_seed=args.seed, runtime_plan=config.plan,
        trace_recorder=AFTraceRecorder(False))
    arrivals = workload(qps, args.requests, args.users, args.seed, args.unique_users)
    dump(out/'arrivals.json', arrivals)
    requests, tokens, batches, queues = [], [], [], []
    sampled = set()
    started = time.monotonic()
    validation = dict(status='passed', profile_scope='First occurrence of each actual batch size only',
        sampled_iterations=0, profile_events=0)

    def progress(iterations, active, pending):
        row = dict(status='running', qps=qps, iterations=iterations,
            completed_requests=sum(r['status']=='complete' for r in requests),
            total_requests=args.requests, generated_tokens=len(tokens),
            active_users=active, pending=pending, simulated_seconds=env.now,
            wall_seconds=time.monotonic()-started)
        dump(out/'progress.json', row)
        print(json.dumps(row), flush=True)

    with gzip.open(out/'sample-operators.trace.json.gz','wt',encoding='utf-8',compresslevel=1) as stream:
        stream.write('{"metadata":{"timing":"SIMULATED analytical_only","scope":"first iteration of each actual batch size"},"traceEvents":[')
        first_event = True

        def serve(reqs, iteration_id):
            nonlocal first_event
            executor.trace.enabled = len(reqs) not in sampled
            iteration = executor.submit(0, reqs, 'decode')
            yield env.process(executor.run_iteration(iteration))
            if executor.trace.enabled:
                events = executor.trace.events
                assert {e.layer_id for e in events if e.layer_id >= 0} == set(range(61))
                by_id = {e.node_id:e for e in events}
                for e in events:
                    assert e.service_duration >= -1e-12 and e.queue_duration >= -1e-12
                    for pred in e.predecessor_ids:
                        assert pred in by_id and by_id[pred].end_at <= e.start_at+1e-9
                    if not first_event:
                        stream.write(',')
                    stream.write(json.dumps(dict(name=e.operation or e.stage or e.event_type,
                        cat=e.event_type, ph='X', pid=1, tid=e.resource_id,
                        ts=e.start_at*1e6, dur=e.service_duration*1e6, args=e.as_dict())))
                    first_event = False
                sampled.add(len(reqs))
                validation['sampled_iterations'] += 1
                validation['profile_events'] += len(events)
                executor.trace.clear()
                stream.flush()
            return {i:iteration.block_completed_at[b.block_id] for b in iteration.blocks for i in b.request_ids}

        schedule(env, arrivals, users=args.users, max_batch=args.max_batch,
            output_tokens=args.output_tokens, context=args.context, serve=serve,
            request_rows=requests, token_rows=tokens, batch_rows=batches, queue_rows=queues,
            busy_policy=args.busy_policy, progress=progress)
        env.run()
        stream.write(']}')
    row = validate_and_summarize(args,qps,arrivals,requests,tokens,batches,queues)
    row['wall_seconds'] = time.monotonic()-started
    for name, data in [('requests',requests),('tokens',tokens),('batches',batches),('queue',queues)]:
        dump(out/f'{name}.json',data)
    with gzip.open(out/'users.trace.json.gz','wt',encoding='utf-8') as stream:
        json.dump(dict(metadata=dict(timing='SIMULATED',scope='All generated tokens'),traceEvents=[
            dict(name='decode token',ph='X',pid=1,tid=t['user'],ts=t['ready_s']*1e6,
                 dur=(t['completion_s']-t['ready_s'])*1e6,args=t) for t in tokens]),stream)
    validation.update(token_accounting='passed', user_serialization='passed',
        sampled_batch_sizes=sorted(sampled), output_tokens=len(tokens))
    for name in ('sample-operators.trace.json.gz','users.trace.json.gz'):
        with gzip.open(out/name,'rb') as stream:
            while stream.read(1024*1024):
                pass
    dump(out/'validation.json',validation)
    dump(out/'summary.json',row)
    dump(out/'progress.json',dict(status='complete',**row))
    print(json.dumps(row),flush=True)
    return row


def plot(output, rows):
    from scripts.plot_v3_user_qps import plot as plot_metrics
    plot_metrics(output, rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qps',type=float,nargs='+',default=[.005,.01,.02,.04,.06,.08,.12,.2,.4])
    parser.add_argument('--users',type=int,default=32)
    parser.add_argument('--requests',type=int,default=64)
    parser.add_argument('--output-tokens',type=int,default=400)
    parser.add_argument('--context',type=int,default=2048)
    parser.add_argument('--a-tp',type=int,default=4)
    parser.add_argument('--f-ep',type=int,default=32)
    parser.add_argument('--max-batch',type=int,default=8)
    parser.add_argument('--af-batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--busy-policy',choices=['queue','drop'],default='queue')
    parser.add_argument('--unique-users',action='store_true',
        help='Assign at most one request to each user; requires requests <= users')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in args.output.glob('qps-*/summary.json')]
        if not rows:
            parser.error('No completed points to plot')
        plot(args.output,rows)
        rows=sorted(rows,key=lambda r:r['qps'])
        dump(args.output/'results.json',rows)
        with (args.output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return
    if any(not math.isfinite(q) or q<=0 for q in args.qps) or min(args.users,args.requests,
        args.output_tokens,args.context,args.a_tp,args.f_ep,args.max_batch,args.af_batch_size)<1:
        parser.error('All rates and counts must be positive and finite')
    if len(set(args.qps))!=len(args.qps):
        parser.error('Duplicate QPS points')
    if args.unique_users and args.requests > args.users:
        parser.error('--unique-users requires --requests <= --users')
    args.output.mkdir(parents=True,exist_ok=True)
    exp=dict(vars(args),output=str(args.output),model='DeepSeek-V3 671B BF16',
        device='a100_sxm_80g',network='nvlink3',timing='analytical_only',
        hypothetical_f_memory_gib=1536,expert_tp=1,router='A',
        arrival_model='Aggregate Poisson arrivals with independent uniformly random user IDs',
        scheduler='At most one active request/user; per-user FIFO queue; round-robin token scheduling; serialized iterations',
        throughput='All emitted tokens / simulated time from 0 through final completion (includes idle and drain)',
        scope='Finite sample from empty queues; prefill assumed complete; no steady-state claim')
    files=sorted({*ROOT.joinpath('TokenSim').rglob('*.py'),*ROOT.joinpath('data').rglob('*.yaml'),
        *ROOT.joinpath('data').rglob('*.json'),Path(__file__),ROOT/'scripts/deepseek_v3_runtime.py',ROOT/'scripts/run_v3_poisson.py'})
    exp['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=args.output/'experiment.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=exp:
        raise ValueError('Experiment changed; use a fresh output directory')
    dump(path,exp)
    rows=[]
    for qps in args.qps:
        path=args.output/f'qps-{qps:g}'/'summary.json'
        rows.append(json.loads(path.read_text(encoding='utf-8')) if path.exists() else run_case(args,qps))
        dump(args.output/'results.json',rows)
        with (args.output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        plot(args.output,rows)


if __name__=='__main__':
    main()
