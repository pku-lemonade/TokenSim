"""Reproducible DSV3 validation; run from the repository root.

python -B -m TokenSim.validation.communication_experiments --suite micro
python -B -m TokenSim.validation.communication_experiments --suite decode
python -B -m TokenSim.validation.communication_experiments --suite service
Outputs stay under TokenSim/validation/results, separate from user experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import sys
import tracemalloc
from types import SimpleNamespace

import simpy

from TokenSim.af.domain_scheduler import DomainScheduler
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.model_executor import ModelAFExecutor
from TokenSim.af.operator_provider import build_operator_table_af_providers
from TokenSim.af.trace import AFTraceRecorder
from TokenSim.comm.traffic import all_to_all_plan, collective_plan, TrafficPhase, Transmission
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.domains import CommunicationDomainConfig, MODEL_VERSION
from TokenSim.errors import ConfigurationError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
SEEDS = (17, 29, 43)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def process_peak_bytes():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('peak_working_set','working_set','peak_paged','paged',
                    'peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        function=ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes=(wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD)
        values=Counters()
        values.cb=ctypes.sizeof(values)
        if not function(kernel.GetCurrentProcess(),ctypes.byref(values),values.cb):
            raise OSError('cannot read process peak working set')
        return values.peak_working_set
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


def configuration(name, block=8, a_tp=1):
    doc = json.loads((ROOT / 'data/clusters/deepseek_v3/1a32f.json').read_text(encoding='utf-8'))
    domains = dict(node_size=8, node_count=5 if name == 'compact' else 9,
                   placement_strategy='dedicated' if name == 'compact' else 'manual',
                   local_bandwidth_bytes_per_s=300e9, inter_tx_bandwidth_bytes_per_s=50e9,
                   inter_rx_bandwidth_bytes_per_s=50e9, local_latency_us=1, inter_latency_us=5,
                   source_id='assumption:dsv3-validation-v1')
    if name != 'compact':
        # Same F mapping and 72-slot inventory; only the A worker moves.
        domains['device_indices'] = {'f-pool:'+str(i): (i//4)*8+i%4 for i in range(32)}
        domains['device_indices']['a-pool:0'] = 64 if name == 'dedicated' else 4
    if a_tp != 1:
        if name != 'compact':
            domains['device_indices'].update({'a-pool:'+str(i):(64 if name=='dedicated' else 4)+i
                                              for i in range(a_tp)})
        doc['num_workers'] = a_tp + 32
        doc['worker_groups'][0]['num_workers'] = a_tp
        a_domain = doc['af']['execution_domains'][0]
        a_domain['parallel']['tensor_parallel_size'] = a_tp
        a_domain['logical_instances'][0]['rank_mapping'] = [dict(
            domain_rank=i,tp_rank=i,pp_rank=0,dp_rank=0,ep_rank=0,
            physical_worker=dict(group_name='a-pool',worker_index=i)) for i in range(a_tp)]
        a_domain['communication_plan'] = [dict(operation=operation,after_stage=stage,
            participants=list(range(a_tp)),network='af-net',resource_group='af-fabric',resource_lane=lane)
            for operation,stage,lane in [('all_reduce','a_compute','a_to_f'),('all_gather','residual','f_to_a')]]
    doc['af'].update(communication_domains=domains, af_block_size=block)
    return doc


def runtime(name='compact', block=8, a_tp=1, doc=None, routing='uniform'):
    hardware = HardwareContext.load(ROOT / 'data')
    model = hardware.model('deepseek-v3-671b-mla')
    if routing != 'uniform':
        from TokenSim.moe.config import RoutingConfig
        model = replace(model,moe=replace(model.moe,routing=RoutingConfig(
            distribution='hot',hot_experts=list(range(8)),hot_expert_fraction=.9)))
    return build_model_runtime(doc or configuration(name, block, a_tp), hardware=hardware,
                               model=model)


def inventory(config):
    domains = config.cluster.af.communication_domains
    mapping = {worker.physical_worker_ref.identity: config.placement.device_index(i)
               for i,worker in enumerate(config.workers)}
    return dict(gpu_total=domains.node_count*domains.node_size,gpu_used=len(mapping),
                empty_slots=domains.node_count*domains.node_size-len(mapping),
                node_count=domains.node_count,placement=mapping,
                domains=asdict(domains),memory=[{key:value for key,value in state.as_dict().items()
                    if key!='components'} for state in config.plan.physical_states])


def validate_kv_reservation(config,context,batch):
    for state in config.plan.physical_states:
        kv_bytes = math.ceil(context/config.plan.block_size)*batch*state.kv_bytes_per_block
        if state.model_bytes+state.protocol_buffer_bytes+kv_bytes>state.capacity_bytes:
            raise ConfigurationError(f'validation KV reservation exceeds {state.physical_identity}: {kv_bytes}')


class CountedEnvironment(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.event_steps = 0

    def step(self):
        self.event_steps += 1
        return super().step()


def executor_for(config, *, seed=17, trace=False, cache=True):
    env = CountedEnvironment()
    compute, transfer = build_operator_table_af_providers(hardware=config.hardware,model=config.model,
        cluster_config=config.cluster,worker_configs=config.workers,placement=config.placement,
        dtype='bf16',fallback='analytical_only')
    compute.cache_enabled = transfer.cache_enabled = cache
    executor = ModelAFExecutor(env,config.cluster,config.workers,compute,transfer,
        config.model.model_id,'bf16',config.plan.total_layers,config.model.hidden_size,
        moe_config=config.model.moe,random_seed=seed,runtime_plan=config.plan,
        trace_recorder=AFTraceRecorder(trace),model=config.model)
    return env, executor, compute, transfer


def request(i, context, phase='decode', step=0):
    return SimpleNamespace(id=i,is_prefill=phase=='prefill',prefill_len=context,
                           prefill_compute_len=context,context_len=context,generation_idx=step)


def route_digest(iteration, accumulator):
    # Excludes iteration/block IDs and placement; records actual expert choices.
    for block in iteration.blocks:
        for snapshot in block.routing_snapshots:
            for assignment in snapshot.assignments:
                accumulator.update(repr((assignment.phase,assignment.layer_id,assignment.request_id,
                    assignment.token_id,assignment.top_k_index,assignment.expert_id)).encode())


def trace_summary(events):
    queue = defaultdict(float)
    wire = defaultdict(float)
    tasks = Counter()
    for event in events:
        tasks[event.layer_id] += 1
        for resource in event.resource_ids:
            if resource.startswith('domain:'):
                queue[resource] += event.queue_duration
        if event.details.get('phase_completion_only'):
            continue
        resource_bytes = event.details.get('resource_wire_bytes')
        if resource_bytes:
            for resource,size in resource_bytes:
                wire[resource] += size
        elif event.resource_ids and event.details.get('wire_bytes') is not None:
            for resource in event.resource_ids:
                wire[resource] += event.details['wire_bytes']
    return dict(domain_queue_s=dict(queue),domain_wire_bytes=dict(wire),tasks_per_layer=dict(tasks),
                task_count=len(events))


def decode_case(config, *, batch=1, context=2048, seed=17, trace=True, cache=True, memory=False):
    validate_kv_reservation(config,context,batch)
    gc.collect()
    if memory:
        tracemalloc.start()
    started, cpu = time.perf_counter(), time.process_time()
    env, executor, compute, transfer = executor_for(config,seed=seed,trace=trace,cache=cache)
    provider_s = time.perf_counter()-started
    built = time.perf_counter()
    iteration = executor.submit(0,[request(i,context) for i in range(batch)],'decode')
    plan_s = time.perf_counter()-built
    routes = hashlib.sha256()
    route_digest(iteration,routes)
    running = time.perf_counter()
    env.process(executor.run_iteration(iteration))
    env.run()
    execution_s = time.perf_counter()-running
    export_started = time.perf_counter()
    # Export exact full rows rather than a sample/truncated task graph.
    exported = json.dumps([event.as_dict() for event in executor.trace.events],default=str)
    export_s = time.perf_counter()-export_started
    peak = tracemalloc.get_traced_memory()[1] if memory else None
    if memory:
        tracemalloc.stop()
    result = dict(batch=batch,context=context,seed=seed,trace=trace,cache=cache,layers=61,
        simulated_s=iteration.makespan,output_tokens=batch,event_steps=env.event_steps,
        provider_wall_s=provider_s,plan_wall_s=plan_s,execution_wall_s=execution_s,
        export_wall_s=export_s,export_bytes=len(exported.encode()),cpu_s=time.process_time()-cpu,
        wall_s=time.perf_counter()-started,peak_tracemalloc_bytes=peak,routing_sha256=routes.hexdigest(),
        process_peak_bytes=process_peak_bytes(),
        cache_hits=compute.estimate_cache_hits+transfer.estimate_cache_hits,
        plan_cache_hits=transfer.plan_cache_hits,**trace_summary(executor.trace.events))
    if trace:
        executor.trace.validate_invariants(iteration.iteration_id)
    return result


def fifo_reference(jobs, capacities):
    env = simpy.Environment()
    scheduler = DomainScheduler(env,capacities)
    result = {}
    def task(name,arrival,size,keys):
        yield env.timeout(arrival)
        ready,start,end = yield from scheduler.run(keys,size/min(capacities[key] for key in keys),(name,))
        result[name] = dict(ready=ready,start=start,end=end,queue=start-ready)
    for name,arrival,size,keys in jobs:
        env.process(task(name,arrival,size,keys))
    env.run()
    return result


def fluid_reference(jobs, capacities):
    """Test-only equal sharing at each resource; not a production mode."""
    pending = sorted(jobs,key=lambda row:(row[1],row[0]))
    active, result, now = {}, {}, 0.0
    while pending or active:
        while pending and pending[0][1] <= now+1e-14:
            name,arrival,size,keys = pending.pop(0)
            active[name] = [size,keys,arrival]
        counts = Counter(key for remaining,keys,_ in active.values() for key in keys)
        rates = {name:min(capacities[key]/counts[key] for key in keys)
                 for name,(_,keys,_) in active.items()}
        finish = min((now+row[0]/rates[name] for name,row in active.items()),default=math.inf)
        following = min(finish,pending[0][1] if pending else math.inf)
        for name,row in active.items():
            row[0] -= (following-now)*rates[name]
        now = following
        for name in list(active):
            if active[name][0] <= 1e-9:
                _,_,arrival = active.pop(name)
                result[name] = dict(ready=arrival,start=arrival,end=now,queue=0.)
    return result


def micro_suite():
    manifest = dict(model_version=MODEL_VERSION,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model='deepseek-v3-671b-mla',layers=61,dense_layers=3,moe_layers=58,hidden=7168,
        experts=256,top_k=8,shared_experts=1,dtype='bf16',seeds=SEEDS,
        assumption='节点级有效聚合带宽；无目标硬件实测；synthetic stateless routing',
        python=sys.version,logical_cpu_count=os.cpu_count(),
        routing_generator_sha256=hashlib.sha256((ROOT/'TokenSim/moe/routing.py').read_bytes()).hexdigest(),
        operator_data_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT/'data/operator_data').glob('*.json'))},configurations={})
    configs = {}
    config_dir = HERE/'configs'
    config_dir.mkdir(exist_ok=True)
    for name in ('compact','dedicated','mixed'):
        config = runtime(name)
        configs[name] = config
        (config_dir/f'dsv3-{name}.json').write_text(json.dumps(config.doc,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest['configurations'][name] = inventory(config)
    for name in ('dedicated','mixed'):
        manifest['configurations']['4a-'+name] = inventory(runtime(name,a_tp=4))
    manifest['coverage'] = {f'V{i}':path for i,path in enumerate((
        'test_communication_domains.py; sensitivity.json', 'fifo_references.json; test_communication_domains.py',
        'fifo_references.json', 'phase_release_reference.json', 'sensitivity.json; formula_delta.json',
        'sensitivity.json:bandwidth', 'placement_decode.json; manifest.json',
        'sensitivity.json:locality; test_communication_domains.py',
        'test_communication_domains.py; tests/test_af_communication_timing.py',
        'cache_trace_equivalence.json; test_communication_domains.py', 'service_load.json',
        'decode_performance.json', 'test_communication_domains.py; full_regression.log'),1)}
    bad = configuration('compact')
    bad['af']['communication_domains'].update(placement_strategy='manual',device_indices={
        'a-pool:0':0,**{'f-pool:'+str(i):i+1 for i in range(32)}})
    try:
        config = runtime(doc=bad)
        executor_for(config)
    except ConfigurationError as exc:
        manifest['invalid_compact_mixed'] = str(exc)
    else:
        raise AssertionError('nonuniform F collective was accepted')
    manifest['model_definition_sha256']=digest(asdict(configs['compact'].model))
    manifest['source_sha256']={name:hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()
        for name in ('comm/traffic.py','hardware/domains.py','hardware/placement.py',
                     'af/domain_scheduler.py','af/estimate_cache.py','af/operator_provider.py','af/executor.py')}
    write('manifest.json',manifest)
    references = {}
    for name,sizes,arrivals in [('equal',(100,100,100),(0,0,0)),('unequal',(100,200,300),(0,0,0)),
                                 ('staggered',(100,200,300),(0,.5,1))]:
        jobs = [(chr(65+i),arrival,size,('local',)) for i,(size,arrival) in enumerate(zip(sizes,arrivals))]
        references[name] = dict(fifo=fifo_reference(jobs,{'local':300}),shared=fluid_reference(jobs,{'local':300}))
    for name,keys in [('same_source',[('tx0','rx1'),('tx0','rx2')]),
                      ('same_target',[('tx0','rx2'),('tx1','rx2')]),
                      ('opposite',[('tx0','rx1'),('tx1','rx0')]),
                      ('disjoint',[('tx0','rx1'),('tx2','rx3')])]:
        caps = {key:50 for group in keys for key in group}
        jobs = [(chr(65+i),0,100,group) for i,group in enumerate(keys)]
        references[name] = dict(fifo=fifo_reference(jobs,caps),shared=fluid_reference(jobs,caps))
    # A phase uses local0 for 1us and inter tx0/rx1 for 10us, concurrently.
    # An unrelated local0 AF message arriving at 2us waits under coarse union.
    config = CommunicationDomainConfig(local_bandwidth_bytes_per_s=100e9,inter_tx_bandwidth_bytes_per_s=100e9,
                                       inter_rx_bandwidth_bytes_per_s=100e9)
    phase = TrafficPhase('fixture','all_to_all',(0,1,2),2,100000,100000,
        ((Transmission(0,1,100000,0,0),Transmission(0,2,1000000,0,1)),))
    timetable = phase.relative_timetable(config)
    references['coarse_vs_fine'] = dict(timetable=timetable,
        coarse_af=fifo_reference([('phase',0,10,('local0','tx0','rx1')),('af',2,1,('local0',))],
                                  dict(local0=1,tx0=1,rx1=1)),
        fine_af=fifo_reference([('local',0,1,('local0',)),('inter',0,10,('tx0','rx1')),('af',2,1,('local0',))],
                                dict(local0=1,tx0=1,rx1=1)))
    write('fifo_references.json',references)
    sensitivity = []
    for participants in (1,2,4,8,32):
        nodes = {rank:rank//8 for rank in range(participants)}
        for size in (0,14336,14336*128,14336*2048):
            for operation in ('all_reduce','all_gather','reduce_scatter'):
                plan = collective_plan(operation,tuple(nodes),size,nodes)
                for alpha in (0,1,5):
                    config = CommunicationDomainConfig(local_latency_us=alpha,inter_latency_us=alpha)
                    sensitivity.append(dict(participants=participants,size=size,operation=operation,
                        alpha_us=alpha,service_s=plan.service(config),wire_bytes=plan.wire_bytes))
    locality = []
    nodes = {rank:rank//8 for rank in range(32)}
    for fraction in (None,.1,.5,.9):
        for hotspot in (1,4):
            plan = all_to_all_plan(tuple(nodes),8*7168*2,nodes,remote_fraction=fraction,load_imbalance=hotspot)
            remote = sum(message.wire_bytes for phase in plan.phases for messages in phase.rounds
                         for message in messages if message.source_node!=message.target_node)
            locality.append(dict(remote_fraction=fraction,hotspot=hotspot,remote_bytes=remote,
                wire_bytes=plan.wire_bytes,service_s=plan.service(CommunicationDomainConfig()),provenance=plan.provenance))
    scans = []
    plan = collective_plan('all_reduce',tuple(nodes),14336*32,nodes)
    for kind,values in [('local',(150e9,300e9,600e9)),('inter',(25e9,50e9,100e9))]:
        for bandwidth in values:
            kwargs = {'local_bandwidth_bytes_per_s':bandwidth} if kind=='local' else {
                'inter_tx_bandwidth_bytes_per_s':bandwidth,'inter_rx_bandwidth_bytes_per_s':bandwidth}
            config = CommunicationDomainConfig(**kwargs)
            scans.append(dict(scan=kind,bandwidth=bandwidth,wire_bytes=plan.wire_bytes,service_s=plan.service(config)))
    write('sensitivity.json',dict(message_cases=sensitivity,locality=locality,bandwidth=scans))
    phase_release_reference()
    print('micro: manifest/configuration/references/sensitivity completed',flush=True)


def decode_suite():
    rows = []
    for a_tp in (1,4):
        for batch,block in ((1,1),(8,8),(32,8)):
            for trace in (False,True):
                for repeat in range(3):
                    arguments=[sys.executable,'-B','-m','TokenSim.validation.communication_experiments',
                        '--suite','decode-case','--batch',str(batch),'--a-tp',str(a_tp)]
                    if trace:
                        arguments.append('--trace')
                    captured=subprocess.run(arguments,cwd=ROOT,capture_output=True,text=True,check=True)
                    row=json.loads(captured.stdout)
                    rows.append(dict(a_tp=a_tp,block=block,repeat=repeat,**row))
                    write('decode_performance.json',rows)
                    print(f'decode A{a_tp} batch={batch} trace={trace} repeat={repeat}: {row["wall_s"]:.3f}s',flush=True)
    paired = []
    for seed in SEEDS:
        for name in ('dedicated','mixed'):
            paired.append(dict(placement=name,**decode_case(runtime(name),batch=8,seed=seed)))
    for seed in SEEDS:
        pair = [row for row in paired if row['seed']==seed]
        assert pair[0]['routing_sha256']==pair[1]['routing_sha256']
    write('placement_decode.json',paired)
    equivalence = []
    for trace,cache in ((True,True),(True,False),(False,True)):
        equivalence.append(decode_case(runtime(),batch=1,trace=trace,cache=cache))
    assert max(row['simulated_s'] for row in equivalence)-min(row['simulated_s'] for row in equivalence)<1e-12
    write('cache_trace_equivalence.json',equivalence)


def full_request(config, *, seed, input_length, output_length=16, request_id=0):
    validate_kv_reservation(config,input_length+output_length,1)
    env,executor,_,_ = executor_for(config,seed=seed,trace=False)
    tokens = []
    step_wall_s = []
    routes = hashlib.sha256()
    cpu,wall = time.process_time(),time.perf_counter()
    # A single admitted request keeps the explicit block workspace budget valid.
    for step in range(output_length):
        step_started = time.perf_counter()
        phase = 'prefill' if step==0 else 'decode'
        iteration = executor.submit(0,[request(request_id,input_length+step,phase,step)],phase)
        route_digest(iteration,routes)
        env.process(executor.run_iteration(iteration))
        env.run()
        tokens.append(env.now)
        step_wall_s.append(time.perf_counter()-step_started)
    assert not executor.resources.domain_scheduler.owners
    assert not executor.resources.domain_scheduler.waiters
    return dict(tokens=tokens,ttft_s=tokens[0],tbt_s=statistics.mean(b-a for a,b in zip(tokens,tokens[1:])) if len(tokens)>1 else None,
        total_s=env.now,routing_sha256=routes.hexdigest(),event_steps=env.event_steps,
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,step_wall_s=step_wall_s,
        dag_template_cache=config.cluster.af.dag_template_cache)


def phase_release_reference():
    from TokenSim.hardware.domains import transfer_resources,transmission_service
    nodes = {rank:rank//8 for rank in range(32)}
    config = CommunicationDomainConfig(local_latency_us=1,inter_latency_us=5)
    plan = collective_plan('all_reduce',tuple(nodes),14336*32,nodes)
    phases = [phase for phase in plan.phases if phase.phase_id in ('local_rs:0','inter_ar:0')]
    cases = [('uniform_collective', phase) for phase in phases]
    # Statistical destination hotspots keep the same global payload and
    # locality, but concentrate more work on the selected remote domain pair.
    for imbalance in (1, 8):
        locality_plan = all_to_all_plan(tuple(nodes),14336*32,nodes,
            remote_fraction=.75,load_imbalance=imbalance)
        phase = next(phase for phase in locality_plan.phases if phase.phase_id=='inter:1:0')
        cases.append((f'all_to_all_imbalance_{imbalance}', phase))
    rows = []
    for case,phase in cases:
        for fine in (False,True):
            env = simpy.Environment()
            scheduler = DomainScheduler(env,phase.resource_ids)
            completed = {}
            def collective():
                if fine:
                    for index,messages in enumerate(phase.rounds):
                        children=[]
                        for message_index,message in enumerate(messages):
                            keys=tuple(resource.identity for resource in transfer_resources(message.source_node,message.target_node))
                            children.append(env.process(scheduler.run(keys,transmission_service(config,
                                message.source_node,message.target_node,message.wire_bytes),(0,index,message_index))))
                        yield env.all_of(children)
                else:
                    yield from scheduler.run(phase.resource_ids,phase.service(config),(0,))
                completed['collective']=env.now
            def af():
                yield env.timeout(phase.service(config)/4)
                ready,start,end=yield from scheduler.run((phase.resource_ids[0],),2e-6,(1,))
                completed['af']=dict(ready=ready,start=start,end=end,queue=start-ready)
            env.process(collective())
            env.process(af())
            env.run()
            rows.append(dict(case=case,phase=phase.phase_id,fine=fine,phase_pure_service_s=phase.service(config),
                             wire_bytes=phase.wire_bytes,**completed))
    write('phase_release_reference.json',rows)


def service_suite():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    rows = []
    # All jobs execute the whole model. Serialized admission is an explicit
    # reference server policy; its external FIFO queue is reported separately.
    for input_length in (128,2048):
        configs = {}
        for name in ('dedicated','mixed'):
            doc=configuration(name,block=input_length)
            # Large one-off prefill shapes have no template reuse in this
            # isolated-session workload. Keep every DAG task while avoiding
            # retention of 58 large, distinct existing DAG recipes.
            if input_length==2048:
                doc['af']['dag_template_cache']=False
            configs[name]=runtime(doc=doc)
            write(f'service_config_{input_length}_{name}.json',doc)
        reference_file=RESULTS/f'service_reference_{input_length}.json'
        first_checkpoint=RESULTS/f'full_request_{input_length}_{SEEDS[0]}_dedicated.json'
        if reference_file.exists() and not first_checkpoint.exists():
            write(first_checkpoint.name,json.loads(reference_file.read_text(encoding='utf-8')))
        missing=[(seed,name) for seed in SEEDS for name in configs
                 if not (RESULTS/f'full_request_{input_length}_{seed}_{name}.json').exists()]
        def execute_sample(seed,name):
            arguments=[sys.executable,'-B','-m','TokenSim.validation.communication_experiments',
                '--suite','full-case','--input-length',str(input_length),'--seed',str(seed),'--placement',name]
            captured=subprocess.run(arguments,cwd=ROOT,capture_output=True,text=True,check=True)
            result=json.loads(captured.stdout)
            write(f'full_request_{input_length}_{seed}_{name}.json',result)
            return seed,name,result
        # Independent SimPy runtimes share no simulated resources. A bounded
        # process pool parallelizes experiments, never communication tasks.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(execute_sample,seed,name) for seed,name in missing]
            for future in as_completed(futures):
                seed,name,result=future.result()
                print(f'full sample {input_length} {seed} {name}: {result["wall_s"]:.2f}s',flush=True)
        reference=json.loads(first_checkpoint.read_text(encoding='utf-8'))
        capacity = 1/reference['total_s']
        write(f'service_reference_{input_length}.json',{**reference,'capacity_requests_s':capacity})
        for seed in SEEDS:
            rng = random.Random(seed)
            # Reused absolute arrival sequences, not each placement's own C.
            gaps = [rng.expovariate(1) for _ in range(3)]
            workloads = {factor:[0]+[sum(gaps[:i+1])/(factor*capacity) for i in range(len(gaps))]
                         for factor in (.3,.7,1.1,2.)}
            service_times = {}
            for name,config in configs.items():
                checkpoint=RESULTS/f'full_request_{input_length}_{seed}_{name}.json'
                if checkpoint.exists():
                    result=json.loads(checkpoint.read_text(encoding='utf-8'))
                else:
                    result=full_request(config,seed=seed,input_length=input_length,request_id=0)
                    write(checkpoint.name,result)
                # Four independent sessions explicitly replay the same frozen
                # assignments with internal request_id=0. Single-flight FCFS
                # drains every resource between sessions, so this replay is
                # exact for this server policy, including internal domain queues.
                # It is not a production whole-request latency cache.
                service_times[name] = [result]*4
                print(f'service input={input_length} seed={seed} {name}: {result["wall_s"]:.2f}s',flush=True)
            assert [r['routing_sha256'] for r in service_times['dedicated']]==[r['routing_sha256'] for r in service_times['mixed']]
            for factor,arrivals in workloads.items():
                for name,requests in service_times.items():
                    available,completed = 0.,[]
                    for arrival,result in zip(arrivals,requests):
                        start=max(arrival,available)
                        available=start+result['total_s']
                        completed.append(dict(arrival=arrival,admission=start,queue_s=start-arrival,
                            first_token=start+result['tokens'][0],end=available,ttft_s=start-arrival+result['tokens'][0],
                            tbt_s=result['tbt_s'],routing_sha256=result['routing_sha256']))
                    window=arrivals[-1]
                    backlog=sum(row['end']>window for row in completed)
                    rows.append(dict(input_length=input_length,output_length=16,layers=61,seed=seed,
                        placement=name,reference_capacity_requests_s=capacity,absolute_rate=factor*capacity,
                        load_factor=factor,admission_policy='single admitted request, FCFS; four independent identical sessions; exact deterministic service replay',
                        arrival_window_s=window,drain_s=max(0,available-window),incomplete_at_window=backlog,
                        incomplete_after_drain=0,throughput_including_drain=4/available,
                        mean_ttft_s=statistics.mean(r['ttft_s'] for r in completed),
                        mean_tbt_s=statistics.mean(r['tbt_s'] for r in completed),requests=completed))
                    write('service_load.json',rows)
            write(f'service_samples_{input_length}_{seed}.json',service_times)


def supplement_suite():
    rows=[]
    for repeat in range(3):
        rows.append(dict(case='context4096',repeat=repeat,
            **decode_case(runtime(),batch=8,context=4096,trace=True)))
        write('supplement.json',rows)
    for a_tp in (1,4):
        for name in ('dedicated','mixed'):
            for seed in SEEDS:
                rows.append(dict(case='paired_placement',a_tp=a_tp,placement=name,
                    **decode_case(runtime(name,a_tp=a_tp),batch=8,seed=seed)))
                write('supplement.json',rows)


def hot_service_suite():
    rows=[]
    reference=json.loads((RESULTS/'service_reference_128.json').read_text(encoding='utf-8'))
    capacity=reference['capacity_requests_s']
    for seed in SEEDS:
        rng=random.Random(seed)
        gaps=[rng.expovariate(1) for _ in range(3)]
        for name in ('dedicated','mixed'):
            config=runtime(name,block=128,routing='hot')
            checkpoint=RESULTS/f'hot_full_request_{seed}_{name}.json'
            result=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else \
                full_request(config,seed=seed,input_length=128)
            write(f'hot_full_request_{seed}_{name}.json',result)
            for factor in (.3,.7,1.1,2.):
                arrivals=[0]+[sum(gaps[:i+1])/(factor*capacity) for i in range(3)]
                available,requests=0.,[]
                for arrival in arrivals:
                    start=max(arrival,available)
                    available=start+result['total_s']
                    requests.append(dict(arrival=arrival,queue_s=start-arrival,ttft_s=start-arrival+result['ttft_s'],
                                         tbt_s=result['tbt_s'],end=available))
                rows.append(dict(seed=seed,placement=name,routing='hot:.9:experts0..7',input_length=128,
                    output_length=16,layers=61,reference_capacity_requests_s=capacity,load_factor=factor,
                    absolute_rate=factor*capacity,requests=requests,drain_s=available-arrivals[-1],
                    incomplete_after_drain=0,routing_sha256=result['routing_sha256'],
                    admission_policy='single-flight FCFS; identical frozen sessions; exact service replay'))
            write('hot_service_load.json',rows)
            print(f'hot service {seed} {name}: {result["wall_s"]:.2f}s',flush=True)
        pair={row['placement']:row['routing_sha256'] for row in rows if row['seed']==seed}
        assert pair['dedicated']==pair['mixed']


def profile_suite():
    import cProfile
    import pstats
    config=runtime()
    profiler=cProfile.Profile(timer=time.process_time)
    profiler.enable()
    result=decode_case(config,batch=8,trace=False)
    profiler.disable()
    categories=defaultdict(lambda:dict(self_cpu_s=0.,calls=0))
    for (filename,line,function),(primitive,calls,self_time,cumulative,callers) in pstats.Stats(profiler).stats.items():
        filename=filename.replace('\\','/').lower()
        if any(name in filename for name in ('estimate_cache.py','operator_provider.py','/operator_data/','/latency/')):
            kind='provider_and_cache'
        elif any(name in filename for name in ('/comm/traffic.py','af/planning.py','af/templates.py','moe/routing.py')):
            kind='plan_and_routing'
        elif any(name in filename for name in ('af/executor.py','af/domain_scheduler.py','/simpy/')):
            kind='event_execution'
        elif any(name in filename for name in ('af/trace.py','/json/','dataclasses.py')):
            kind='trace_and_serialization'
        else:
            kind='other'
        categories[kind]['self_cpu_s']+=self_time
        categories[kind]['calls']+=calls
    write('profile_attribution.json',dict(result=result,categories=dict(categories),
        scope='单独 CPU profiler 样本，含 profiler 开销；各类别 self time 不重复相加，未替换 provider 方法或改变缓存 fast path'))
    print('CPU attribution completed',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=('micro','decode','decode-case','full-case','service','probe','supplement','hot-service','profile'),default='micro')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--a-tp',type=int,default=1)
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--placement',choices=('dedicated','mixed'),default='dedicated')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--input-length',type=int,default=128)
    parser.add_argument('--output-length',type=int,default=16)
    parser.add_argument('--dag-cache-off',action='store_true')
    args=parser.parse_args()
    if args.suite=='decode-case':
        result=decode_case(runtime(block=1 if args.batch==1 else 8,a_tp=args.a_tp),batch=args.batch,trace=args.trace)
        print(json.dumps(result))
    elif args.suite=='full-case':
        doc=configuration(args.placement,block=args.input_length)
        if args.input_length==2048:
            doc['af']['dag_template_cache']=False
        result=full_request(runtime(doc=doc),seed=args.seed,input_length=args.input_length)
        write(f'full_request_{args.input_length}_{args.seed}_{args.placement}.json',result)
        print(json.dumps(result))
    elif args.suite=='probe':
        doc=configuration('dedicated',block=args.input_length)
        if args.dag_cache_off:
            doc['af']['dag_template_cache']=False
        result=full_request(runtime(doc=doc),seed=17,
                            input_length=args.input_length,output_length=args.output_length)
        write(f'probe_{args.input_length}_{args.output_length}.json',result)
        print(json.dumps(result,indent=2),flush=True)
    else:
        {'micro':micro_suite,'decode':decode_suite,'service':service_suite,
         'supplement':supplement_suite,'hot-service':hot_service_suite,'profile':profile_suite}[args.suite]()
