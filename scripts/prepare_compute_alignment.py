"""Prepare a new remote GPU0/3 experiment from the known-working AFD driver."""
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    base = Path('/home/youwei/autodl-tmp/af-workspace')
    source = base / 'profiling/afd-alignment-20261006'
    root = base / 'profiling/afd-compute-gpu03-20261006'
    root.mkdir(exist_ok=False)
    for name in ('run_curve.py', 'attention.py'):
        code = (source / name).read_text()
        code = code.replace('GPU-8df5e331-8f9a-3cdc-af30-42b94f908018',
                            'GPU-96dd5c70-92fb-45b3-1ebf-82de94078d4d')
        code = code.replace('a_nvidia_smi_index=2', 'a_nvidia_smi_index=0')
        code = code.replace('17639', '17739').replace('17640', '17740').replace('18693', '18793')
        code = code.replace('indices 2 (A), 3 (F)', 'indices 0 (A), 3 (F)')
        code = code.replace('GPU 2 and 3 only', 'GPU 0 and 3 only')
        (root / name).write_text(code)
    manifest = dict(scope='decode compute only; no network or engine-overhead fitting',
                    source_driver=str(source), gpu_indices=[0,3], held_out_batch=4,
                    training_batches=[1,8], parameters_fitted=False)
    manifest['inventory'] = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv'], text=True)
    manifest['topology'] = subprocess.check_output(['nvidia-smi','topo','-m'], text=True)
    files = [root/'run_curve.py', root/'attention.py', base/'models/qwen3-moe-tiny-random/config.json',
             base/'src/afd-plugin/afd_plugin/model_executor/models/qwen3_moe.py']
    manifest['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(root)


if __name__ == '__main__':
    main()
