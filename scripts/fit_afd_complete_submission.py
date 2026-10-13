"""Ready-input CPU submission including audited nonblocking CUDA APIs."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
from scripts.report_afd_api_audit import QUERY_APIS,LAUNCH_APIS


def mapping(role):
    if role=='f':return {'compute_f_moe_block_us':[(f'model.layers.{i}.mlp',) for i in range(2)]}
    return {'compute_a_embedding_us':[('model.embed_tokens',)],'compute_a_final_norm_us':[('model.norm',)],
        'compute_a_lm_head_us':[('lm_head',)],'compute_a_token_selection_us':[('token_selection',)],
        'compute_a_attention_us':[(f'model.layers.{i}.input_layernorm',f'model.layers.{i}.self_attn') for i in range(2)],
        'compute_a_post_norm_us':[(f'model.layers.{i}.post_attention_layernorm',) for i in range(2)]}


def qualify(original,audit):
    if original.get('timing_scope')!='host_service_excluding_cuda_api_union' or audit.get('timing_scope')!='diagnostic_api_audit_not_fit':
        raise ValueError('original and audit scopes required')
    if not original['prior_sync_outside_timer'] or not audit['prior_sync_outside_timer']:
        raise ValueError('prior device dependencies must be outside timer')
    if original['role']!=audit['role'] or original['batch']!=audit['batch'] or original['gpu_uuid']!=audit['gpu_uuid']:
        raise ValueError('native API audit workload/device mismatch')
    if set(original['groups'])!=set(audit['groups']):raise ValueError('native group mismatch')
    for name,rows in original['groups'].items():
        if original['sources'][name]['sha256']!=audit['sources'][name]['sha256']:
            raise ValueError('native function source drift')
        if {r['cuda_api_calls'] for r in rows}!={r['cuda_api_calls'] for r in audit['groups'][name]}:
            raise ValueError('CUDA API invocation count drift')
        if len(rows)!=200 or len(audit['groups'][name])!=200:raise ValueError('incomplete native window')
        for row in audit['groups'][name]:
            for api in row['outer_cuda_apis']:
                if api['name'] not in QUERY_APIS|LAUNCH_APIS|{'cudaMemcpyAsync'}:
                    raise ValueError('blocking or unknown CUDA API; refusing submission fit')
                if api['name']=='cudaMemcpyAsync' and api.get('copy_kinds')!=8:
                    raise ValueError('only audited asynchronous device-to-device copies admitted')
            if not math.isclose(sum(r['duration_us'] for r in row['outer_cuda_apis']),row['excluded_cuda_api_us'],abs_tol=1e-7):
                raise ValueError('API union mismatch')
        for row in rows:
            if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0 for k in ('service_us','wall_us','excluded_cuda_api_us')):
                raise ValueError('invalid native submission sample')
            if not math.isclose(row['service_us']+row['excluded_cuda_api_us'],row['wall_us'],abs_tol=1e-7):
                raise ValueError('original interval accounting mismatch')


def fit(samples,audit_root,base,device_source):
    values={};source_hashes={}
    for role in ('a','f'):
        for b in (1,8):
            audit_path=audit_root/f'b{b}-w0'/f'{role}-compute-cpu-samples.json'
            audit=json.loads(audit_path.read_text(encoding='utf-8'))
            source_hashes[str(audit_path)]=hashlib.sha256(audit_path.read_bytes()).hexdigest()
            for w in range(3):
                path=samples/f'{role}-b{b}-w{w}.json';d=json.loads(path.read_text(encoding='utf-8'))
                if d['batch']!=b or d['window']!=w or d['role']!=role:raise ValueError('training case mismatch')
                qualify(d,audit);source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                for field,calls in mapping(role).items():
                    group_values=[[sum(d['groups'][name][i]['wall_us'] for name in call) for i in range(200)] for call in calls]
                    values.setdefault(field,[]).append((b,w,group_values))
    profile=copy.deepcopy(base)
    profile.update(schema_version=4,profile_id='pkua100-complete-submission-20261012-v1',
        timing_scope='host_submission_service',excludes_device_launch=False,
        device_service_source_id=device_source,
        submission_audit=dict(source=source_hashes,source_hashes_verified=True,call_counts_verified=True,
            blocking_apis_absent=True,copy_kinds=[3]))
    profile['evidence'].update(heldout_batches=[7],validation_status='pending')
    report={}
    for field,windows in values.items():
        medians=[statistics.mean(statistics.median(v) for v in groups) for b,w,groups in windows]
        constant=statistics.mean(medians)
        flat=sorted(v for b,w,groups in windows for group in groups for v in group)
        profile['costs'][field]['fixed_us']=constant
        profile['evidence']['quantiles_us'][field]=dict(p50=statistics.median(flat),p95=flat[math.ceil(.95*len(flat))-1])
        profile['evidence']['residual_us']=max(profile['evidence']['residual_us'],max(abs(v-constant) for v in medians))
        report[field]=dict(fixed_us=constant,previous_framework_only_us=base['costs'][field]['fixed_us'],
            window_medians_us=medians,samples=len(flat))
    profile['evidence']['source']+=';audited CPU API inclusive; source and API counts matched to preserved training windows'
    return profile,dict(costs=report,source_sha256=source_hashes,
        scope='caller CPU entry to return; prior work excluded; no device completion wait',
        limitations=['CUPTI callbacks perturb CPU service; historical no-CUPTI controls retained',
            'call groups are aggregate CPU costs, not exact kernel enqueue overlap',
            'must replace old assumed launch coverage with independently fitted GPU-only service'])


def main():
    p=argparse.ArgumentParser()
    for name in ('samples','audit-root','base-profile','device-profile','output','report'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source='device_service:'+hashlib.sha256(a.device_profile.read_bytes()).hexdigest()
    d,r=fit(a.samples,a.audit_root,json.loads(a.base_profile.read_text(encoding='utf-8')),source)
    for path,doc in ((a.output,d),(a.report,r)):
        with path.open('x',encoding='utf-8') as f:json.dump(doc,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
