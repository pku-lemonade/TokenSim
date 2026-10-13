"""Prepare isolated native capture and three independent timing windows. No jobs launched."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'driver source drift at {old!r}')
    return source.replace(old,new,1)


def prepare(base, target, hooks, audit):
    base, target = base.resolve(), target.resolve()
    source = base/'profiling/afd-dbo-fixed-20261007'
    plugin = base/'src/afd-plugin-dbo-cache-20261007'
    if target == base or base not in target.parents or target.exists():
        raise ValueError('target must be a new directory below the AF workspace')
    hashes = json.loads(audit.read_text())['source_hashes']
    for name, expected in hashes.items():
        if sha(plugin/name) != expected:
            raise ValueError(f'plugin source drift: {name}')
    attention=(source/'attention.py').read_text()
    runner=(source/'run_curve.py').read_text()
    # Validate all insertion points before creating the output.
    changed_attention=replace_once(attention, '    nv.nvmlInit()',
        "    if os.environ.get('AFD_CAPTURE_DIR'):\n        (Path(os.environ['AFD_CAPTURE_DIR'])/'capture.enabled').touch(exist_ok=False)\n    nv.nvmlInit()")
    changed_attention=replace_once(changed_attention,
        "    assert 'afd-plugin-dbo-cache-20261007' in actual_plugin.parts, actual_plugin",
        "    assert actual_plugin.is_relative_to(Path(os.environ['AFD_ISOLATED_PLUGIN']).resolve()), actual_plugin")
    changed_runner=replace_once(runner, "str(base/'src/afd-plugin-dbo-cache-20261007')",
        "os.environ['AFD_ISOLATED_PLUGIN']")
    guard="    assert not any(uuid in apps for uuid in uuids), 'Selected GPUs have other processes; stop without interference'"
    changed_runner=replace_once(changed_runner, guard, guard+'''
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    selected = {row.split(',')[0].strip():row.split(',')[1:] for row in inventory.splitlines()}
    for uuid in uuids:
        memory, utilization = map(lambda value: float(value.strip()), selected[uuid])
        if memory > 100 or utilization > 5:
            raise RuntimeError('Target GPU is not idle; no workers launched')''')
    for code in (changed_attention,changed_runner):
        ast.parse(code)
    target.mkdir(parents=True)
    isolated=target/'plugin'
    shutil.copytree(plugin,isolated,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copy2(hooks,isolated/'afd_capture_hooks.py')
    for name, snippet in (
        ('attention_model_runner.py', '\nfrom afd_capture_hooks import install_attention\ninstall_attention(AFDAttentionModelRunner)\n'),
        ('ffn_model_runner.py', '\nfrom afd_capture_hooks import install_ffn\nfrom vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter\ninstall_ffn(GPUFFNModelRunner, FusedMoERouter)\n')):
        path=isolated/'afd_plugin/v1/worker'/name
        content=path.read_text()+snippet
        ast.parse(content)
        path.write_text(content)
    cases=[]
    for mode, windows in (('capture',1),('timing',3)):
        for batch in (1,8,4):
            for window in range(windows):
                directory=target/f'{mode}-b{batch}-w{window}'
                directory.mkdir()
                (directory/'run_curve.py').write_text(changed_runner)
                (directory/'attention.py').write_text(changed_attention)
                env={'AF_WORKSPACE':str(base),'AFD_ISOLATED_PLUGIN':str(isolated)}
                if mode=='capture':
                    env.update(AFD_CAPTURE_DIR=str(directory), AFD_CAPTURE_BATCH=str(batch), AFD_CAPTURE_STEPS='31')
                cases.append(dict(mode=mode,batch=batch,window=window,environment=env,
                    command=[str(base/'envs/afd026/bin/python'),str(directory/'run_curve.py'),'--concurrency',str(batch)],
                    driver_hashes={name:sha(directory/name) for name in ('run_curve.py','attention.py')},
                    result=str(directory/f'c{batch}/result.json')))
    manifest=dict(schema_version=1,source_hashes=hashes,
        driver_hashes={name:sha(source/name) for name in ('run_curve.py','attention.py')},
        isolated_hashes={name:sha(isolated/name) for name in hashes}, hooks_sha256=sha(hooks), cases=cases,
        status='prepared_not_executed', capture_scope='intrusive, no timing calibration',
        execution_order='timing batches 1/8; freeze profile; timing batch4; capture separately',
        heldout_requires_frozen_profile=True)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Confirm original source unchanged even if preparation later fails elsewhere.
    if any(sha(plugin/name)!=expected for name,expected in hashes.items()):
        raise RuntimeError('original plugin changed during preparation')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--hooks',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    manifest=prepare(args.base,args.target,args.hooks,args.audit)
    print(json.dumps(dict(status=manifest['status'],cases=len(manifest['cases']),target=str(args.target))))


if __name__=='__main__': main()
