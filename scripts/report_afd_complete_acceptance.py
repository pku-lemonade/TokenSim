"""Join independent same-scope evidence; do not infer component success from step."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from TokenSim.af.transport_profile import TransportServiceProfile
from scripts.fit_afd_device_fixed_service import device_totals


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def frozen_profile_sha(bundle, root, filename, key):
    relative='docs/plans/'+filename
    path=root/relative
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=bundle['implementation_sha256'][relative] or load(path)!=bundle[key]:
        raise ValueError('frozen profile bytes or content mismatch')
    return digest


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','steps','components','transport','transport-candidate','device-native','replay-root','source-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();bundle=load(a.bundle);step=load(a.steps);host=load(a.components);transport=load(a.transport)
    if bundle['schema_version']!=4 or not step['frozen_hashes_verified'] or not step['all_steps_pass']:
        raise ValueError('frozen complete-cost step validation missing')
    if {r['batch'] for r in step['results']}!={1,8,4,7} or not next(r for r in step['results'] if r['batch']==7)['heldout']:
        raise ValueError('unseen batch7 required')
    if host['comparison_profile_sha256']!=frozen_profile_sha(bundle,a.source_root,'afd-complete-host-profile-20261012.json','host_profile') or not host['all_component_order_pass']:
        raise ValueError('independent host/operation profile evidence mismatch')
    if {r['field'] for r in host['results']}!=set(bundle['host_profile']['costs'])|set(bundle['operation_profile']['operations']):
        raise ValueError('fixed cost coverage incomplete')
    if (not transport['device_service_order_pass'] or not transport['peer_delay_stable']
        or transport['profile_sha256']!=hashlib.sha256(a.transport_candidate.read_bytes()).hexdigest()
        or load(a.transport_candidate)['records']!=bundle['transport_profile']['records']):
        raise ValueError('transport candidate numbers changed or validation failed')
    native=load(a.device_native/'device-summary.json');contract=load(a.device_native/'contract.json')
    if contract['heldout_bundle_sha256']!=hashlib.sha256(a.bundle.read_bytes()).hexdigest():
        raise ValueError('GPU heldout contract mismatch')
    if len(native)!=3 or {(r['batch'],r['window']) for r in native}!={(7,w) for w in range(3)}:
        raise ValueError('three independent batch7 GPU windows required')
    model=load(a.replay_root/'b7/tokensim-b7.json');device=[]
    if model['device_service_profile_sha256']!=frozen_profile_sha(bundle,a.source_root,'afd-device-service-profile-20261012.json','device_service_profile'):
        raise ValueError('device calibration differs from frozen bundle')
    for row in native:
        for role,actual in device_totals(row).items():
            predicted=model['decode']['compute_service_by_endpoint_us'][role]
            if min(actual,predicted)<=0:raise ValueError('positive GPU services required')
            ratio=max(actual,predicted)/min(actual,predicted)
            device.append(dict(role=role,window=row['window'],native_us=actual,model_us=predicted,ratio=ratio,order_pass=ratio<10))
    profile=TransportServiceProfile(bundle['transport_profile'],environment=bundle['transport_profile']['environment'],
        domains=bundle['transport_profile']['domains'],device_indices=bundle['transport_profile']['device_indices'])
    for batch in (1,8,4,7):
        events=load(a.replay_root/f'b{batch}'/f'tokensim-b{batch}-decode-events.json')
        wires=[e for e in events if e['event_type'] in ('transfer','control_transfer')]
        if len(wires)!=31*6:raise ValueError('wire count mismatch')
        for e in wires:
            dtype=e['details'].get('wire_dtype','bf16')
            source,target=(2,3) if e['direction']=='a_to_f' else (3,2)
            expected=profile.estimate_us(source=source,target=target,backend=e['details']['transport_backend'],dtype=dtype,size=e['payload_bytes'])
            if e['source_id']!=profile.document['profile_id'] or not math.isclose(e['service_duration']*1e6,expected,abs_tol=1e-7):
                raise ValueError('transport profile not actually charged')
    all_pass=all(r['order_pass'] for r in device)
    if not all_pass:raise ValueError('GPU heldout service missed quantity target')
    report=dict(scope='tiny Qwen3 MoE, BF16 eager, GPU2/3 same-node 1A1F decode',
        acceptance_target='positive same-scope max/min < 10; causal oracle for waiting',
        scoped_alignment_passed=all_pass,step_results=step['results'],device_heldout=device,
        host_operation_max_ratio=max(max(r['window_ratios']) for r in host['results']),
        transport_max_ratio=max(r['ratio'] for r in transport['results']),
        frozen_bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
        independent_component_validation_kind=host['comparison_kind'],
        limitations=['quantity agreement does not imply small percentage error',
            'CPU/runtime groups and GPU fixed-service coefficient are aggregate approximations',
            'CUPTI and cross-device timing uncertainty retained; transport is device service, not physical wire-only time',
            'no prefill, cross-node, DSV3 or multi-inflight validation'])
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(scoped_alignment_passed=all_pass,max_gpu_ratio=max(r['ratio'] for r in device))))


if __name__=='__main__':main()
