"""Capture pre-migration estimates, timelines and simulator overhead once."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import simpy

from tests.test_af_executor import af_test_hardware
from tests.test_af_parallel_planning import _cluster
from tests.test_af_communication_timing import ZeroCompute
from TokenSim.af import AFExecutor, TopologyAFTransferProvider
from TokenSim.hardware.placement import build_topology_placement


def capture():
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "predecessor": "integrate-deepseek-v3-core-runtime:d24df24",
        "scope": "minimal two-worker synthetic baseline; not a hardware measurement",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (root/'comm/collectives.py', root/'af/executor.py',
                                       root/'af/operator_provider.py', root/'af/resources.py')},
        "cases": [],
    }
    for mode in ('flat_link', 'physical'):
        for trace in (False, True):
            samples = []
            for _ in range(3):
                cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1)
                cluster.af.transfer_topology_mode = mode
                cluster.af.trace_enabled = trace
                workers = cluster.workers()
                hardware = af_test_hardware()
                placement = build_topology_placement(cluster, workers, hardware)
                provider = TopologyAFTransferProvider(hardware, cluster, workers,
                                                     placement, fallback='analytical_only')
                env = simpy.Environment()
                executor = AFExecutor(env, cluster, workers, ZeroCompute(), provider,
                                      'TestModel', 'bf16', 1, 16)
                request = SimpleNamespace(id=0, is_prefill=True, prefill_len=1,
                                          prefill_compute_len=1, context_len=1)
                iteration = executor.submit(0, [request], 'prefill')
                process_started, wall_started = time.process_time(), time.perf_counter()
                env.process(executor.run_iteration(iteration))
                env.run()
                samples.append({'cpu_s': time.process_time()-process_started,
                                'wall_s': time.perf_counter()-wall_started})
            manifest['cases'].append({
                'mode': mode, 'trace': trace, 'samples': samples,
                'configuration': {'workers': 2, 'hidden_size': 16, 'tokens': 1,
                                  'dtype': 'bf16', 'host_timing': cluster.af.communication_timing.enabled},
                'makespan_s': iteration.makespan,
                'events': [{'operation': event.operation, 'resources': event.resource_ids,
                            'ready': event.ready_at, 'start': event.start_at, 'end': event.end_at,
                            'payload_bytes': event.payload_bytes}
                           for event in executor.trace.events],
                'provider': provider.describe(),
            })
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(capture(), target, indent=2)
