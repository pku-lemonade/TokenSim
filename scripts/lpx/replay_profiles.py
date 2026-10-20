"""Capture and cache sampled whole-model native operator graphs."""

from dataclasses import asdict, dataclass
import fcntl
from functools import lru_cache
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from time import perf_counter

import simpy
import numpy as np

from TokenSim.af.serving.service import AFService
from TokenSim.af.serving.timing import namespace_resource
from TokenSim.af.trace.occupancy import occupied_until
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def source_identity():
    digest = hashlib.sha256()
    paths = [
        *ROOT.joinpath("TokenSim").rglob("*.py"),
        *ROOT.joinpath("scripts/lpx").rglob("*.py"),
        Path(__file__).with_name("replay_engine.cpp"),
    ]
    for directory in ("devices", "models", "topologies"):
        paths.extend((ROOT / "data" / directory).rglob("*.yaml"))
    paths.extend(
        path for path in (ROOT / "data/operator_data").rglob("*") if path.is_file()
    )
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeProfile:
    nodes: tuple
    metadata: dict

    def verify(self):
        kernel = ReplayKernel()
        try:
            profile = kernel.add(self.nodes)
            kernel.submit(profile, 0)
            result = kernel.advance()
            if result is None or not math.isclose(
                result.end,
                self.metadata["native_duration_s"],
                rel_tol=1e-7,
                abs_tol=1e-10,
            ):
                actual = None if result is None else result.end
                raise SimulationStateError(
                    f"native profile reconstruction differs: {actual} versus {self.metadata['native_duration_s']}"
                )
        finally:
            kernel.close()


def extract_profile(events, namespace, native_duration, metadata):
    ids = {event.node_id: index for index, event in enumerate(events)}
    if len(ids) != len(events):
        raise SimulationStateError("sample profile has duplicate native nodes")
    hosts = [
        i for i, event in enumerate(events) if event.operation == "cuda_graph_submit"
    ]
    ranks = {
        index: rank
        for rank, index in enumerate(
            sorted(range(len(events)), key=lambda i: (events[i].start_at, i))
        )
    }
    nodes = []
    for index, event in enumerate(events):
        predecessors = [ids[name] for name in event.predecessor_ids]
        if not predecessors and hosts and index not in hosts:
            predecessors = list(hosts)
        parent_end = max((events[parent].end_at for parent in predecessors), default=0)
        delay = max(0.0, event.ready_at - parent_end)
        duration = event.end_at - event.start_at
        occupancy = occupied_until(event) - event.start_at
        keys = event.resource_ids or (event.resource_id,)
        nodes.append(
            ReplayNode(
                tuple(predecessors),
                tuple(namespace_resource(namespace, key) for key in keys),
                namespace_resource(namespace, event.resource_id),
                delay,
                duration,
                occupancy,
                ranks[index],
            )
        )
    result = NativeProfile(
        tuple(nodes),
        {
            **metadata,
            "native_duration_s": native_duration,
            "native_events": len(events),
        },
    )
    result.verify()
    return result


def capture_profile(
    runtime,
    namespace,
    *,
    batch,
    queries,
    phase,
    emit_logits,
    context,
    dp_rank,
    seed,
    sample,
):
    started = perf_counter()
    env = simpy.Environment()
    service = AFService(env, runtime, seed=seed, retain_trace=True)
    sessions = [
        service.admit(
            index + sample * batch,
            dp_rank,
            maximum_context=context,
            current_context=context - queries,
        )
        for index in range(batch)
    ]
    for session in sessions:
        session.generated = 1 + sample if phase == "decode" else 0
    env.process(
        service.forward(
            sessions, phase=phase, query_tokens=queries, emit_logits=emit_logits
        )
    )
    env.run()
    result = extract_profile(
        service.trace.events,
        namespace,
        env.now,
        dict(
            phase=phase,
            batch=batch,
            queries=queries,
            context=context,
            emit_logits=emit_logits,
            dp_rank=dp_rank,
            seed=seed,
            sample=sample,
            source_sha256=source_identity(),
            host_capture_s=perf_counter() - started,
        ),
    )
    service.trace.clear()
    del service, env, sessions
    gc.collect()
    return result


def cached_profile(runtime, namespace, cache, **settings):
    identity = dict(
        runtime_versions=dict(
            python=list(sys.version_info[:3]),
            numpy=np.__version__,
            simpy=simpy.__version__,
        ),
        source=source_identity(),
        model=runtime.model.to_dict(),
        devices={
            name: runtime.hardware.device(name).to_dict()
            for name in sorted({worker.hardware for worker in runtime.workers})
        },
        links={
            name: asdict(runtime.hardware.link(name))
            for name in sorted(set(runtime.cluster.networks.values()))
        },
        configuration=runtime.doc,
        namespace=namespace,
        settings=settings,
    )
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.json.gz"
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with gzip.open(path, "rt") as stream:
                saved = json.load(stream)
            if saved["input_sha256"] != key:
                raise ConfigurationError(
                    "native replay cache input fingerprint mismatch"
                )
            data = saved["nodes"]
            if (
                hashlib.sha256(
                    json.dumps(data, separators=(",", ":")).encode()
                ).hexdigest()
                != saved["body_sha256"]
            ):
                raise ConfigurationError(
                    "native replay cache body fingerprint mismatch"
                )
            result = NativeProfile(
                tuple(
                    ReplayNode(tuple(row[0]), tuple(row[1]), *row[2:]) for row in data
                ),
                saved["metadata"],
            )
            result.verify()
            return result
        result = capture_profile(runtime, namespace, **settings)
        data = [
            [
                node.predecessors,
                node.resources,
                node.primary,
                node.delay,
                node.duration,
                node.occupancy,
                node.rank,
            ]
            for node in result.nodes
        ]
        body_hash = hashlib.sha256(
            json.dumps(data, separators=(",", ":")).encode()
        ).hexdigest()
        result = NativeProfile(
            result.nodes,
            {
                **result.metadata,
                "service": namespace,
                "model_id": runtime.model.model_id,
                "input_sha256": key,
                "node_body_sha256": body_hash,
            },
        )
        saved = dict(
            input_sha256=key,
            body_sha256=body_hash,
            nodes=data,
            metadata=result.metadata,
        )
        with tempfile.NamedTemporaryFile(
            dir=cache, suffix=".gz", delete=False
        ) as temporary:
            name = temporary.name
        try:
            with gzip.open(name, "wt", compresslevel=3) as stream:
                json.dump(saved, stream, separators=(",", ":"), allow_nan=False)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        return result
