"""Read retained same-window CPU/GPU traces on the server; export small audits."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def role_audit(path):
    doc=json.loads(path.read_text());events=doc['traceEvents']
    apis={e.get('args',{}).get('correlation'):e for e in events
          if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    ops=defaultdict(list)
    for e in events:
        if e.get('cat') in ('cpu_op','user_annotation'):
            ops[e.get('args',{}).get('External id')].append(e)
    kernels=[e for e in events if e.get('cat')=='kernel']
    matched=Counter();missing=Counter();launch_gaps=[];negative=[];symbol_recovered=0
    for k in kernels:
        api=apis.get(k.get('args',{}).get('correlation'))
        cpu=ops.get(k.get('args',{}).get('External id'),[])
        if api is None:
            missing['no_runtime_correlation:'+k['name']]+=1
        elif not cpu and k['name']=='_compute_slot_mapping_kernel':
            # Direct Triton launch has no enclosing torch CPU-op annotation.
            # CUDA correlation still identifies its launch; the audited kernel
            # symbol belongs to block-table input preparation, not attention.
            matched['audited_triton_input_prepare:slot_mapping']+=1
            symbol_recovered+=1
        elif not cpu:
            missing['no_cpu_external_id:'+k['name']]+=1
        else:
            matched[min(cpu,key=lambda x:x.get('dur',0))['name']]+=1
            gap=k['ts']-api['ts'];launch_gaps.append(gap)
            if gap<-.01:negative.append(gap)
    messages=sorted([e for e in events if e.get('name') in ('vllm::afd_p2p_send','vllm::afd_p2p_recv')],key=lambda x:x['ts'])
    simplified=[dict(name=e['name'],start_us=e['ts'],end_us=e['ts']+e['dur'],
        shape=e['args']['Input Dims'][0],external_id=e['args'].get('External id')) for e in messages]
    steps=[e['dur'] for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')]
    result=dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_time_ns=doc['baseTimeNanoseconds'],
        kernel_count=len(kernels),matched_kernel_count=sum(matched.values()),cpu_operator_kernel_counts=dict(matched),
        unmatched_kernel_reasons=dict(missing),symbol_recovered_no_cpu_op_annotation=symbol_recovered,
        negative_launch_order_count=len(negative),
        profiler_step_median_us=statistics.median(steps[1:]),profiled_steps=len(steps),
        launch_to_kernel_gap_scope='includes device queue/wait, not fixed overhead')
    return result,simplified


def pair_messages(a,f,batch):
    pairs=[];unmatched=0
    for direction,send_role,recv_role in [('a_to_f',a,f),('f_to_a',f,a)]:
        sends=[r for r in send_role if r['name'].endswith('_send')]
        recvs=[r for r in recv_role if r['name'].endswith('_recv')]
        assert sum(r['shape'][0]==batch*128 for r in sends)==2
        assert sum(r['shape'][0]==batch*128 for r in recvs)==2
        # The profile windows start at different loop positions. Anchor at
        # their unique prefill burst, then pair FIFO by the reviewed 1A1F wire.
        si=next(i for i,r in enumerate(sends) if r['shape'][0]==batch*128)
        ri=next(i for i,r in enumerate(recvs) if r['shape'][0]==batch*128)
        count=min(len(sends)-si,len(recvs)-ri)
        unmatched+=si+ri+abs((len(sends)-si)-(len(recvs)-ri))
        for ordinal,(s,r) in enumerate(zip(sends[si:si+count],recvs[ri:ri+count])):
            assert s['shape']==r['shape']
            pairs.append(dict(direction=direction,phase='prefill' if ordinal<2 else 'decode',
                decode_step=(ordinal-2)//2 if ordinal>=2 else None,layer=ordinal%2,
                shape=s['shape'],send_start_us=s['start_us'],send_return_us=s['end_us'],
                receive_start_us=r['start_us'],receive_return_us=r['end_us']))
    return dict(identity_contract='unique prefill burst anchor + FIFO + two layers + exact shape; single inflight',
        aligned_messages=len(pairs),window_boundary_unmatched_calls=unmatched,pairs=pairs)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=[]
    for batch in (1,8):
        roles={};messages={}
        for role in ('attention','ffn'):
            roles[role],messages[role]=role_audit(next((a.root/f'c{batch}'/role).glob('*.pt.trace.json')))
        assert roles['attention']['base_time_ns']==roles['ffn']['base_time_ns']
        calls=json.loads((a.root/f'c{batch}/calls.json').read_text())
        unprofiled=[step*1000 for call in calls for step in call['step_ms'][1:]]
        results.append(dict(batch=batch,roles=roles,messages=pair_messages(messages['attention'],messages['ffn'],batch),
            profiler_perturbation=dict(unprofiled_decode_samples=len(unprofiled),
                unprofiled_decode_median_us=statistics.median(unprofiled),
                attention_profiled_over_unprofiled=roles['attention']['profiler_step_median_us']/statistics.median(unprofiled),
                scope='same run configuration, separate profiled and unprofiled phases; no per-function correction inferred')))
    report=dict(scope='retained DBO-fixed paired CPU/GPU profiles; diagnostic only, no service fit',
        clock_contract='same host Kineto baseTimeNanoseconds; use common epoch, never align by equal ProfilerStep index',
        limitations=['cross-process skew below API timing resolution not independently bounded',
                    'profiled durations include profiler perturbation; no fixed cost imported',
                    'boundary calls outside common capture remain unmatched'],results=results)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps([{ 'batch':r['batch'],'roles':r['roles'],
                      'messages':{k:v for k,v in r['messages'].items() if k!='pairs'}} for r in results]))


if __name__=='__main__':main()
