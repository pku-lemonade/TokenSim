"""Check frozen constants on independent batch6; no fitting or parameter writes."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def positive(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('finite positive service measurement required')
    return value


def compare(constants,windows):
    if set(constants)!=set(windows):
        raise ValueError(f'coverage mismatch: {set(constants)^set(windows)}')
    result=[]
    for field,constant in constants.items():
        positive(constant)
        if set(windows[field])!={0,1,2}:
            raise ValueError('three independent windows required: '+field)
        medians=[]; counts=[];p95=[]
        for w in range(3):
            values=sorted(positive(v) for v in windows[field][w])
            if len(values)<50:raise ValueError('insufficient component samples')
            medians.append(statistics.median(values));counts.append(len(values))
            p95.append(values[math.ceil(.95*len(values))-1])
        ratios=[max(constant,v)/min(constant,v) for v in medians]
        result.append(dict(field=field,frozen_us=constant,window_p50_us=medians,
            window_p95_us=p95,samples_per_window=counts,window_ratios=ratios,
            residual_us=[v-constant for v in medians],order_pass=all(v<10 for v in ratios)))
    return result


def report(root,bundle_path,comparison_profile=None):
    bundle=load(bundle_path);contract=load(root/'frozen-validation-contract.json')
    digest=hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if contract['bundle_sha256']!=digest or contract['validation_batch']!=6 or contract['used_for_fit'] is not False:
        raise ValueError('independent frozen validation contract mismatch')
    host=load(comparison_profile) if comparison_profile else bundle['host_profile']
    if comparison_profile and (host['schema_version']!=4 or host['evidence']['training_batches']!=[1,8]):
        raise ValueError('retrospective independent comparison requires training-only v4 profile')
    constants={k:v['fixed_us'] for k,v in host['costs'].items()}
    constants.update({k:v['fixed_us'] for k,v in bundle['operation_profile']['operations'].items()})
    windows=defaultdict(dict);sources={};source_groups=defaultdict(set)
    def read(path):
        d=load(path);sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return d
    def add(name,w,values):
        if w in windows[name]:raise ValueError('duplicate component window')
        windows[name][w]=values
    def native(mode,name,w):
        directory=root/mode/f'b6-w{w}'
        if mode=='engine':directory=directory/'c6'
        d=read(directory/name)
        if d['batch']!=6 or d['window']!=w or d.get('profiler_enabled') is not False:
            raise ValueError('native component window mismatch')
        source_groups[(mode,name)].add(json.dumps(d.get('sources',d.get('source_sha256')),sort_keys=True))
        return d
    for w in range(3):
        for mode,name,field in [('inputs','prepare-samples.json','a_input_prepare'),
                ('metadata','a-metadata-samples.json','a_attention_metadata'),
                ('metadata','f-metadata-samples.json','f_dp_metadata'),
                ('sampled_tokens','sampled-token-samples.json','sampled_token_readback')]:
            d=native(mode,name,w)
            if d['timing_scope']!='operation_service' or d['external_wait_excluded'] is not True:
                raise ValueError('operation boundary mismatch')
            if len(d['samples'])!=200:raise ValueError('incomplete operation samples')
            add(field,w,[r['service_us'] for r in d['samples']])
        d=native('bookkeeping','bookkeeping-samples.json',w)
        if not d['readback_excluded'] or d['timing_scope']!='host_service':
            raise ValueError('bookkeeping scope mismatch')
        if any(not math.isclose(r['wall_us']-r['excluded_readback_us'],r['service_us'],abs_tol=1e-7) for r in d['samples']):
            raise ValueError('nested readback accounting mismatch')
        add('engine_runner_bookkeeping_us',w,[r['service_us'] for r in d['samples']])
        for role in ('a','f'):
            d=native('compute_cpu',f'{role}-compute-cpu-samples.json',w)
            if d['timing_scope']!='host_service_excluding_cuda_api_union' or not d['prior_sync_outside_timer']:
                raise ValueError('compute host scope mismatch')
            groups=d['groups']
            mapping=({'compute_a_embedding_us':[('model.embed_tokens',)],
                'compute_a_final_norm_us':[('model.norm',)],'compute_a_lm_head_us':[('lm_head',)],
                'compute_a_token_selection_us':[('token_selection',)],
                'compute_a_attention_us':[(f'model.layers.{i}.input_layernorm',f'model.layers.{i}.self_attn') for i in range(2)],
                'compute_a_post_norm_us':[(f'model.layers.{i}.post_attention_layernorm',) for i in range(2)]}
                if role=='a' else {'compute_f_moe_block_us':[(f'model.layers.{i}.mlp',) for i in range(2)]})
            for rows in groups.values():
                if len(rows)!=200 or any(not math.isclose(r['wall_us']-r['excluded_cuda_api_us'],r['service_us'],abs_tol=1e-7) for r in rows):
                    raise ValueError('compute API union mismatch')
            for field,calls in mapping.items():
                key='wall_us' if comparison_profile else 'service_us'
                add(field,w,[statistics.mean(sum(groups[n][i][key] for n in call) for call in calls) for i in range(200)])
        d=native('engine','engine-cpu-samples.json',w)
        if d['contains_wait'] is not False or d['timing_scope']!='host_service':
            raise ValueError('engine scope mismatch')
        add('engine_step_schedule_us',w,[r['schedule'] for r in d['samples']])
        add('engine_output_process_us',w,[r['update_from_output']+r['process_outputs'] for r in d['samples']])
        for row in read(root/'staging'/f'window-{w}.json'):
            if row['batch']!=6 or row['window']!=w or not row['prior_gpu_work_synchronized'] or row['profiler_enabled']:
                raise ValueError('staging boundary mismatch')
            add(row['operation'],w,row['wall_us'])
        d=read(root/f'codec-w{w}.json')
        if d['cuda_initialized']:raise ValueError('codec must be CPU-only')
        by_field=defaultdict(list)
        for r in d['samples']:
            if r['batch']!=6 or r['contains_wait'] or r['profiler_enabled']:
                raise ValueError('codec boundary mismatch')
            by_field[r['cost_field']].append(r['duration_us'])
        for field,values in by_field.items():add(field,w,values)
        for backend in ('torch','pynccl'):
            by_field=defaultdict(list)
            for rank in (0,1):
                d=read(root/f'{backend}-w{w}'/f'rank{rank}.json')
                if d['profiler_enabled'] or d['blocking_wait'] or not d['completion_sync_outside_timer']:
                    raise ValueError('submission boundary mismatch')
                for row in d['rows']:
                    if row['batch']!=6 or row['window']!=w:raise ValueError('submission case mismatch')
                    early=(row['action']=='send' and row['delayed_role']=='receiver') or (row['action']=='receive' and row['delayed_role']=='sender')
                    if early and any(r['returned_before_peer_entered'] is not True for r in row['samples']):
                        raise ValueError('external peer waiting not excluded')
                    if row['delay_ms']==0:
                        field=f"{row['action']}_{'torch_distributed_nccl' if backend=='torch' else backend}_{row['direction']}_us"
                        by_field[field].extend(r['service_us'] for r in row['samples'])
            for field,values in by_field.items():add(field,w,values)
    if any(len(v)!=1 or 'null' in v for v in source_groups.values()):
        raise ValueError('native function source drift')
    rows=compare(constants,windows)
    return dict(validation_batch=6,used_for_fit=False,bundle_sha256=digest,results=rows,
        comparison_profile_sha256=hashlib.sha256(comparison_profile.read_bytes()).hexdigest() if comparison_profile else None,
        comparison_kind='retrospective_independent_not_blind' if comparison_profile else 'frozen_before_observation',
        all_component_order_pass=all(r['order_pass'] for r in rows),source_sha256=sources,
        scope=('25 host/operation constants including CPU submission APIs; transport/device validated separately'
               if comparison_profile else '25 frozen host/operation constants; excludes uncalibrated CUDA API and transport'),
        full_alignment_passed=False)


def main():
    p=argparse.ArgumentParser()
    for name in ('root','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compare-host-profile',type=Path)
    a=p.parse_args();result=report(a.root,a.bundle,a.compare_host_profile)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('validation_batch','all_component_order_pass')}))


if __name__=='__main__':main()
