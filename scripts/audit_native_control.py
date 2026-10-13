"""CPU-only source audit: execute the reference serializer, no plugin import."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
relative = [
    'afd_plugin/connectors/metadata.py', 'afd_plugin/connectors/gpu/p2p.py',
    'afd_plugin/v1/worker/attention_metadata.py', 'afd_plugin/v1/worker/attention_model_runner.py',
    'afd_plugin/v1/worker/ffn_model_runner.py', 'afd_plugin/v1/worker/ffn_worker.py',
    'afd_plugin/model_executor/models/qwen3_moe.py',
]
source = (root / relative[0]).read_text()
tree = ast.parse(source)
names = {'encode_control_payload', '_to_int'}
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(selected) == 2
namespace = dict(json=json, AFDControlPayload=object, Any=object, cast=lambda _, x: x)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(root / relative[0]), 'exec'), namespace)
examples = []
for batch in (1, 8, 4, 16):
    tokens = SimpleNamespace(tolist=lambda b=batch: [b])
    payload = SimpleNamespace(dp_metadata_list={0: SimpleNamespace(
        num_tokens_across_dp_cpu=tokens, max_tokens_across_dp_cpu=batch)},
        is_graph_capturing=False, is_warmup=False, is_graph_replaying=False, is_profile=False)
    wire = namespace['encode_control_payload'](payload)
    examples.append(dict(batch=batch, length_bytes=8, payload_bytes=len(wire), payload_utf8=wire.decode()))
print(json.dumps(dict(source_root=str(root), source_hashes={
    p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in relative},
    source_lines={n.name: n.lineno for n in selected}, examples=examples), indent=2))
