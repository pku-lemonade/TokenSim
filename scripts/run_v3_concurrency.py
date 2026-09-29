"""Closed-loop V3 decode: fixed user slots per point, round-robin token scheduling."""
from __future__ import annotations

import argparse
from collections import deque
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simpy
from TokenSim.af import AFTraceRecorder, build_operator_table_af_providers
from scripts.deepseek_v3_runtime import configuration
from TokenSim.af.model_executor import ModelAFExecutor


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def closed_loop(env, *, users, max_batch, output_tokens, warmup_rounds,
                measure_rounds, context, serve, tokens, batches, lifecycles, windows):
    """Each persistent user issues a fresh prefilled request after completion.

    Iterations are serialized; AF blocks within each iteration may overlap.
    Each full round gives each user exactly one token, including N > batch cap.
    A user's next token cannot enter an iteration before its last token completes.
    """
    next_id = 0
    queue = deque()

    def new_request(slot, ready):
        nonlocal next_id
        req = SimpleNamespace(id=next_id, slot=slot, is_prefill=False,
            prefill_len=context, context_len=context, generation_idx=0,
            ready=ready, arrival=ready, last_token=None)
        next_id += 1
        lifecycles.append(dict(request_id=req.id, user=slot, arrival_s=ready))
        return req

    for slot in range(users):
        queue.append(new_request(slot, 0.0))
    for round_id in range(warmup_rounds + measure_rounds):
        if round_id == warmup_rounds:
            windows['start_s'] = env.now
        remaining = users
        while remaining:
            requests = [queue.popleft() for _ in range(min(max_batch, remaining))]
            remaining -= len(requests)
            started = env.now
            completions = yield env.process(serve(requests, round_id))
            batches.append(dict(round=round_id, start_s=started, end_s=env.now,
                users=[r.slot for r in requests], request_ids=[r.id for r in requests]))
            for req in requests:
                end = completions[req.id]
                assert req.ready <= started + 1e-9 and started <= end <= env.now + 1e-9
                tokens.append(dict(user=req.slot, request_id=req.id,
                    token_index=req.generation_idx, context_len=req.context_len,
                    round=round_id, measured=round_id >= warmup_rounds,
                    ready_s=req.ready, admitted_s=started, completion_s=end,
                    scheduler_wait_ms=(started-req.ready)*1000,
                    service_ms=(end-started)*1000,
                    itl_ms=None if req.last_token is None else (end-req.last_token)*1000))
                req.generation_idx += 1
                if req.generation_idx == output_tokens:
                    lifecycles[req.id]['completion_s'] = end
                    # Logical replacement is ready at own-block completion.
                    # Like all requests, it can enter only the next iteration.
                    req = new_request(req.slot, end)
                else:
                    req.context_len = context + req.generation_idx
                    req.ready = end
                    req.last_token = end
                queue.append(req)
        assert len(queue) == users and len({r.slot for r in queue}) == users
    windows['end_s'] = env.now


def run_case(args, users):
    out = args.output/f'n{users}'
    out.mkdir(parents=True, exist_ok=True)
    config = configuration(full_model=True, a_tp=args.a_tp, f_tp=32,
        af_batch_size=8, device_id='a100_sxm_80g', network='nvlink3',
        hypothetical_f_memory_gib=1536)
    dump(out/'cluster.json', config.doc)
    dump(out/'runtime-plan.json', config.plan.summary())
    dump(out/'hardware.json', dict(devices={w.hardware:config.hardware.device(w.hardware).to_dict()
        for w in config.workers}, network=config.hardware.link('nvlink3').__dict__))
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(hardware=config.hardware,
        model=config.model, cluster_config=config.cluster, worker_configs=config.workers,
        placement=config.placement, dtype='bf16', fallback='analytical_only')
    executor = ModelAFExecutor(env, config.cluster, config.workers, compute, transfer,
        config.model.model_id, 'bf16', 61, config.model.hidden_size,
        moe_config=config.model.moe, random_seed=42, runtime_plan=config.plan,
        trace_recorder=AFTraceRecorder(False))
    sampled = False
    validation = dict(status='passed', profile_scope='First measured iteration only; token trace covers all iterations')

    def serve(requests, round_id):
        nonlocal sampled
        if requests[0].slot == 0 and round_id % 10 == 0:
            progress = dict(status='running', a_tp=args.a_tp, users=users,
                completed_rounds=round_id,
                total_rounds=args.warmup_rounds + args.measure_rounds,
                simulated_seconds=env.now, wall_seconds=time.monotonic()-started)
            dump(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        executor.trace.enabled = round_id == args.warmup_rounds and not sampled
        iteration = executor.submit(0, requests, 'decode')
        yield env.process(executor.run_iteration(iteration))
        if executor.trace.enabled:
            sampled = True
            events = executor.trace.events
            assert {e.layer_id for e in events if e.layer_id >= 0} == set(range(61))
            by_id = {e.node_id:e for e in events}
            with gzip.open(out/'sample-operators.trace.json.gz','wt',encoding='utf-8',compresslevel=1) as stream:
                stream.write('{"metadata":{"timing":"SIMULATED analytical_only; first measured iteration only"},"traceEvents":[')
                for index, event in enumerate(events):
                    for pred in event.predecessor_ids:
                        assert pred in by_id and by_id[pred].end_at <= event.start_at+1e-9
                    assert event.service_duration >= -1e-12 and event.queue_duration >= -1e-12
                    if index: stream.write(',')
                    stream.write(json.dumps(dict(name=event.operation or event.stage or event.event_type,
                        cat=event.event_type, ph='X', pid=1, tid=event.resource_id,
                        ts=event.start_at*1e6,dur=event.service_duration*1e6,args=event.as_dict())))
                stream.write(']}')
            validation.update(profile_events=len(events), layers=61, iteration=iteration.iteration_id)
            executor.trace.clear()
        return {request_id:iteration.block_completed_at[block.block_id]
            for block in iteration.blocks for request_id in block.request_ids}

    tokens, batches, lifecycles, windows = [], [], [], {}
    started = time.monotonic()
    env.process(closed_loop(env, users=users,max_batch=args.max_batch,
        output_tokens=args.output_tokens,warmup_rounds=args.warmup_rounds,
        measure_rounds=args.measure_rounds,context=2048,serve=serve,
        tokens=tokens,batches=batches,lifecycles=lifecycles,windows=windows))
    env.run()
    duration = windows['end_s']-windows['start_s']
    measured = [t for t in tokens if t['measured']]
    assert len(measured) == users*args.measure_rounds
    assert all(windows['start_s'] < t['completion_s'] <= windows['end_s'] for t in measured)
    assert all(len(b['users']) <= args.max_batch for b in batches)
    # Cross-request/user causality is checked independently of the sampled operator DAG.
    for user in range(users):
        previous = -1.0
        for token in (t for t in tokens if t['user']==user):
            assert token['ready_s'] >= previous - 1e-9
            assert token['completion_s'] > previous
            previous = token['completion_s']
    user_rates = [sum(t['user']==u for t in measured)/duration for u in range(users)]
    total_tps = len(measured)/duration
    tps_user = sum(user_rates)/users
    assert abs(total_tps-users*tps_user) < 1e-8
    itls = [t['itl_ms'] for t in measured if t['itl_ms'] is not None]
    row = dict(a_tp=args.a_tp,f_ep=32,users=users,total_tps=total_tps,tps_user=tps_user,
        **windows,measurement_seconds=duration,measured_tokens=len(measured),
        mean_user_tps=tps_user,min_user_tps=min(user_rates),max_user_tps=max(user_rates),
        mean_itl_ms=sum(itls)/len(itls) if itls else None,
        mean_scheduler_wait_ms=sum(t['scheduler_wait_ms'] for t in measured)/len(measured),
        mean_service_ms=sum(t['service_ms'] for t in measured)/len(measured),
        max_actual_batch=max(len(b['users']) for b in batches),
        completed_requests=sum('completion_s' in r for r in lifecycles),
        wall_seconds=time.monotonic()-started)
    dump(out/'tokens.json',tokens)
    dump(out/'batches.json',batches)
    dump(out/'requests.json',lifecycles)
    dump(out/'per-user.json',[dict(user=u,output_tokens=args.measure_rounds,tps=rate) for u,rate in enumerate(user_rates)])
    dump(out/'validation.json',dict(**validation,token_causality='passed',window_identity='passed'))
    token_events=[dict(name='decode token',ph='X',pid=1,tid=t['user'],
        ts=t['ready_s']*1e6,dur=(t['completion_s']-t['ready_s'])*1e6,args=t) for t in tokens]
    with gzip.open(out/'users.trace.json.gz','wt',encoding='utf-8') as stream:
        json.dump(dict(traceEvents=token_events,metadata=dict(timing='SIMULATED',**windows)),stream)
    dump(out/'summary.json',row)
    dump(out/'progress.json',dict(status='complete',a_tp=args.a_tp,users=users,
        completed_rounds=args.warmup_rounds+args.measure_rounds,
        total_rounds=args.warmup_rounds+args.measure_rounds,wall_seconds=row['wall_seconds']))
    print(json.dumps(row),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-tp',type=int,default=2)
    parser.add_argument('--users',type=int,nargs='+',default=[1,2,4,8,16,32,64,128])
    parser.add_argument('--max-batch',type=int,default=32)
    parser.add_argument('--output-tokens',type=int,default=4)
    parser.add_argument('--warmup-rounds',type=int,default=1)
    parser.add_argument('--measure-rounds',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.a_tp,*args.users,args.max_batch,args.output_tokens,args.warmup_rounds,args.measure_rounds)<1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    exp=dict(vars(args),output=str(args.output),model='DeepSeek-V3 671B BF16',layers=61,
        f_ep=32,expert_tp=1,af_batch_size=8,cached_context=2048,
        device='a100_sxm_80g',network='nvlink3',hypothetical_f_memory_gib=1536,
        timing='analytical_only',router='A',scheduler='Round robin one token/user/round; max batch cap',
        x='Mean per-user output tokens / common measurement duration; includes all scheduling delays',
        y='Total output tokens / same duration; no power normalization',
        limitations='Finite simulated sample, no steady-state guarantee; requests already prefilled; full operator trace sampled for first measured iteration only')
    for source in ('scripts/run_v3_concurrency.py','scripts/deepseek_v3_runtime.py','TokenSim/af/executor.py'):
        exp.setdefault('source_sha256',{})[source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    path=args.output/'experiment.json'
    if path.exists() and json.loads(path.read_text())!=exp:
        raise ValueError('Experiment changed; use new output directory')
    dump(path,exp)
    rows=[]
    for users in args.users:
        path=args.output/f'n{users}'/'summary.json'
        rows.append(json.loads(path.read_text()) if path.exists() else run_case(args,users))
        dump(args.output/'results.json',rows)
    with (args.output/'results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
