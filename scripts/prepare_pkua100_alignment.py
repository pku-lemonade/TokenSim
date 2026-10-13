"""Prepare an isolated rerun of the existing server-side 1A1F benchmark."""
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def main():
    base = Path('/home/youwei/autodl-tmp/af-workspace')
    source = base / 'profiling/curve-1a1f-20260915'
    output = base / 'profiling/afd-alignment-20261006'
    output.mkdir(exist_ok=True)
    manifest = {'scope': 'complete tiny Qwen3 MoE engine, not full DeepSeek V3',
                'source_driver': str(source), 'baseline': 'existing server plugin snapshot',
                'model': str(base / 'models/qwen3-moe-tiny-random'), 'files': {}}
    for name in ('vllm', 'vllm-afd-plugin', 'torch', 'transformers'):
        try:
            manifest[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            manifest[name] = 'not installed'
    for name in ('run_curve.py', 'attention.py'):
        text = (source / name).read_text()
        if name == 'run_curve.py':
            text = text.replace('(17539,17540,18593)', '(17639,17640,18693)')
            text = text.replace('port=17539', 'port=17639').replace('port=17539,', 'port=17639,')
            text = text.replace("'18593'", "'18693'")
        target = output / name
        if target.exists() and target.read_text() != text:
            raise RuntimeError(f'Existing driver differs: {target}')
        target.write_text(text)
        manifest['files'][name] = hashlib.sha256(text.encode()).hexdigest()
    for relative in ('afd_plugin/model_executor/models/qwen3_moe.py',
                     'afd_plugin/connectors/gpu/p2p.py'):
        path = base / 'src/afd-plugin' / relative
        if path.exists():
            manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['gpu_inventory'] = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
        '--format=csv,noheader'], text=True)
    manifest['topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    config = base / 'models/qwen3-moe-tiny-random/config.json'
    manifest['model_config'] = json.loads(config.read_text())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'directory': str(output), 'versions': {
        k: manifest[k] for k in ('vllm', 'vllm-afd-plugin', 'torch')}}))


if __name__ == '__main__':
    main()
