"""Six isolated batch1/8 native windows; installed plugin remains unchanged."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise ValueError('source drift: '+old)
    return code.replace(old,new,1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['inputs','metadata','sampled_tokens','bookkeeping','compute_cpu'], default='inputs')
    parser.add_argument('--cases', nargs='+', help='Selected batch:window pairs, e.g. 8:1 1:2 8:2; output must still be new')
    parser.add_argument('--validation-batch', type=int, choices=[6],
                        help='Independent component validation only; never passed to training fitters')
    args = parser.parse_args()
    cases = [(b,w) for w in range(3) for b in (1,8)]
    if args.validation_batch is not None:
        if args.cases:
            parser.error('validation batch cannot be combined with training cases')
        cases = [(args.validation_batch,w) for w in range(3)]
    if args.cases:
        selected = [tuple(map(int, item.split(':'))) for item in args.cases]
        if len(set(selected)) != len(selected) or any(item not in cases for item in selected):
            raise ValueError('cases must be unique batch1/8 window0/1/2 pairs')
        cases = selected
    root, base = args.output.resolve(), args.base.resolve()
    if root.exists() or base not in root.parents:
        raise ValueError('output must be a new directory within workspace')
    original = base/'src/afd-plugin-dbo-cache-20261007'
    source = base/'profiling/afd-native-collection-20261007-v1/timing-b1-w0'
    plugin_file = 'afd_plugin/v1/worker/attention_model_runner.py'
    digest = hashlib.sha256((original/plugin_file).read_bytes()).hexdigest()
    driver = (source/'attention.py').read_text()
    driver = replace_once(driver, '    nv.nvmlInit()',
        "    (Path(os.environ['AFD_PREPARE_DIR'])/'measure.enabled').touch(exist_ok=False)\n    nv.nvmlInit()")
    # Disable warmup profiling too, so measured windows carry no profiler state.
    driver = '\n'.join(line for line in driver.splitlines() if 'profile incomplete' not in line)+'\n'
    runner = (source/'run_curve.py').read_text()
    runner = runner.replace('dict(ENABLE=1,WAIT=32', 'dict(ENABLE=0,WAIT=32')
    runner = '\n'.join(line for line in runner.splitlines()
                       if "assert len(list(directory.glob('*/*.pt.trace.json')))==2" not in line)+'\n'
    if 'dict(ENABLE=0,WAIT=32' not in runner:
        raise ValueError('profiler source drift')
    ast.parse(driver); ast.parse(runner)
    root.mkdir()
    plugin = root/'plugin'
    shutil.copytree(original,plugin,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copy2(Path(__file__).with_name('afd_prepare_hooks.py'),plugin/'afd_prepare_hooks.py')
    if args.mode=='compute_cpu':
        shutil.copy2(Path(__file__).with_name('afd_compute_cpu_hooks.py'),plugin/'afd_compute_cpu_hooks.py')
    path = plugin/plugin_file
    path.write_text(path.read_text()+'\nfrom afd_prepare_hooks import install\ninstall(AFDAttentionModelRunner)\n')
    ffn_path = plugin/'afd_plugin/v1/worker/ffn_model_runner.py'
    ffn_path.write_text(ffn_path.read_text()+'\nfrom afd_prepare_hooks import install_ffn\ninstall_ffn(GPUFFNModelRunner)\n')
    manifest = dict(source_sha256=digest, cases=[])
    for batch, window in cases:
        directory = root/f'b{batch}-w{window}'
        directory.mkdir()
        (directory/'attention.py').write_text(driver)
        (directory/'run_curve.py').write_text(runner)
        env = {**os.environ, 'AF_WORKSPACE':str(base), 'AFD_ISOLATED_PLUGIN':str(plugin),
               'AFD_PREPARE_DIR':str(directory), 'AFD_PREPARE_BATCH':str(batch),
               'AFD_PREPARE_WINDOW':str(window),'AFD_PREPARE_MODE':args.mode}
        for name in list(env):
            if name.startswith('AFD_CAPTURE_'):
                env.pop(name)
        with (directory/'collection.log').open('x') as log:
            subprocess.run([str(base/'envs/afd026/bin/python'),str(directory/'run_curve.py'),
                            '--concurrency',str(batch)], env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        names = (['prepare-samples.json'] if args.mode=='inputs' else
                 ['sampled-token-samples.json'] if args.mode=='sampled_tokens' else
                 ['bookkeeping-samples.json'] if args.mode=='bookkeeping' else
                 ['a-compute-cpu-samples.json','f-compute-cpu-samples.json'] if args.mode=='compute_cpu' else
                 ['a-metadata-samples.json','f-metadata-samples.json'])
        for name in names:
            samples = json.loads((directory/name).read_text())
            if args.mode=='compute_cpu':
                assert all(len(group)==200 for group in samples['groups'].values())
            else:
                assert len(samples['samples'])==200
        result = json.loads((directory/f'c{batch}/result.json').read_text())
        assert result['repeat_token_min_match']==1
        manifest['cases'].append(dict(batch=batch,window=window,directory=str(directory),
                                       repeat_token_min_match=result['repeat_token_min_match']))
        print(json.dumps(manifest['cases'][-1]),flush=True)
    assert hashlib.sha256((original/plugin_file).read_bytes()).hexdigest()==digest
    with (root/'manifest.json').open('x') as stream:
        json.dump(manifest,stream,indent=2)


if __name__ == '__main__':
    main()
