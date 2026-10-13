"""Fit native CPU call groups after removing all caller CUDA API intervals."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from TokenSim.af.calibration import HostOverheadProfile,EXTENDED_HOST_COVERAGE,content_fingerprint


def native_groups(role):
    if role=='f':return {f'model.layers.{i}.mlp' for i in range(2)}
    return {'model.embed_tokens','model.norm','lm_head','token_selection'} | {
        f'model.layers.{i}.{name}' for i in range(2) for name in ('input_layernorm','self_attn','post_attention_layernorm')}


def fit(documents,base):
    windows={};sources={};probes=[]
    for doc in documents:
        role,batch,window=doc['role'],doc['batch'],doc['window']
        key=(role,batch,window)
        if role not in ('a','f') or type(batch) is not int or type(window) is not int or batch not in (1,8) or window not in range(3) or key in windows:
            raise ValueError('unique role/training batch/window required')
        if (doc.get('timing_scope')!='host_service_excluding_cuda_api_union'
                or doc.get('prior_sync_outside_timer') is not True
                or doc.get('cuda_runtime_and_driver_callbacks') is not True
                or doc.get('profiler_enabled') is not False
                or doc.get('gpu_uuid')!=base['environment']['gpu_pair'].split(',')[0 if role=='a' else 1]):
            raise ValueError('invalid CPU/API exclusion boundary')
        expected=native_groups(role)
        if set(doc['groups'])!=expected or set(doc['sources'])!=expected:
            raise ValueError('native groups missing or unknown')
        sources.setdefault(role,set()).add(content_fingerprint(doc['sources']))
        probe=doc['empty_probe_us']
        if len(probe)!=100 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in probe):
            raise ValueError('invalid empty probe evidence')
        probes.append(statistics.median(probe))
        for rows in doc['groups'].values():
            if len(rows)!=200:raise ValueError('200 invocations per group required')
            for row in rows:
                if (type(row['cuda_api_calls']) is not int or row['cuda_api_calls']<=0
                    or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0
                        for k in ('service_us','wall_us','excluded_cuda_api_us'))
                    or not math.isclose(row['wall_us'],row['service_us']+row['excluded_cuda_api_us'],abs_tol=1e-7)):
                    raise ValueError('unqualified CUDA API union')
        windows[key]=doc['groups']
    if len(windows)!=12 or any(len(v)!=1 for v in sources.values()):
        raise ValueError('three windows per batch/role and stable sources required')
    costs={}
    for (role,b,w),groups in windows.items():
        mapping = ({'compute_a_embedding_us': [('model.embed_tokens',)],
                    'compute_a_final_norm_us':[('model.norm',)],'compute_a_lm_head_us':[('lm_head',)],
                    'compute_a_token_selection_us':[('token_selection',)],
                    'compute_a_attention_us':[(f'model.layers.{i}.input_layernorm',f'model.layers.{i}.self_attn') for i in range(2)],
                    'compute_a_post_norm_us':[(f'model.layers.{i}.post_attention_layernorm',) for i in range(2)]}
                   if role=='a' else {'compute_f_moe_block_us':[(f'model.layers.{i}.mlp',) for i in range(2)]})
        for field,calls in mapping.items():
            values=[[sum(groups[name][i]['service_us'] for name in call) for i in range(200)] for call in calls]
            costs.setdefault(field,[]).append(dict(batch=b,window=w,values=values))
    result=copy.deepcopy(base)
    result['profile_id']='pkua100-native-fixed-costs-20261009-v1'
    evidence=result['evidence'];evidence['validation_status']='pending'
    evidence['source']+=';compute-cpu-excludes-all-CUDA-API:'+content_fingerprint(documents)
    report=dict(scope='native framework CPU outside CUDA runtime/driver API union',
        empty_probe_window_medians_us=probes,
        limitations=['empty probe does not measure all CUPTI callback perturbation',
                    'CUDA runtime/driver intervals excluded; existing device floor remains an analytical assumption',
                    'call-group CPU before device is an eager DAG approximation, not exact stream replay'],costs={})
    for field,rows in costs.items():
        medians=[statistics.mean(statistics.median(group) for group in row['values']) for row in rows]
        constant=statistics.mean(medians)
        if constant<=max(probes)*2:raise ValueError('probe too large relative to CPU service')
        values=sorted(v for row in rows for group in row['values'] for v in group)
        quantiles=dict(p50=statistics.median(values),p95=values[math.ceil(len(values)*.95)-1])
        result['costs'][field]=dict(fixed_us=constant,per_token_us=0,coverage=[EXTENDED_HOST_COVERAGE[field]])
        evidence['quantiles_us'][field]=quantiles
        evidence['sample_count']+=len(values)
        evidence['residual_us']=max(evidence['residual_us'],max(abs(v-constant) for v in medians))
        report['costs'][field]=dict(fixed_us=constant,window_medians_us=medians,quantiles_us=quantiles,
            invocations_per_step=2 if field in ('compute_a_attention_us','compute_a_post_norm_us','compute_f_moe_block_us') else 1)
    HostOverheadProfile(result,expected_runtime=base['runtime_fingerprint'],expected_environment=base['environment'],explicit_costs=())
    return result,report


def main():
    p=argparse.ArgumentParser();p.add_argument('samples',type=Path)
    p.add_argument('--base-profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    result,report=fit([json.loads(f.read_text()) for f in sorted(a.samples.glob('*-b*-w*.json'))],
        json.loads(a.base_profile.read_text(encoding='utf-8')))
    for path,data in ((a.output,result),(a.report,report)):
        with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False)
    print(json.dumps(report))


if __name__=='__main__':main()
