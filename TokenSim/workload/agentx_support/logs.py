"""Streaming evidence for native replay and physical cache pressure."""

from __future__ import annotations

import json
from pathlib import Path


class ReplayLogs:
    def __init__(self, directory, engine):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.files = {
            name: (directory / f"{name}.jsonl").open("w")
            for name in ("events", "requests", "cache_windows")
        }
        configurations, indices, workers = [], {}, []
        for worker in engine.request_workers:
            cache = worker.cache_config.to_dict()
            identity = json.dumps(cache, sort_keys=True)
            if identity not in indices:
                indices[identity] = len(configurations)
                configurations.append(cache)
            workers.append({"worker_id": worker.id, "configuration": indices[identity]})
        (directory / "cache_configuration.json").write_text(
            json.dumps({"configurations": configurations, "workers": workers}, indent=2)
            + "\n"
        )

    def write(self, name, record):
        self.files[name].write(json.dumps(record, separators=(",", ":")) + "\n")

    def snapshot(self, env, replay):
        workers = []
        for worker in self.engine.request_workers:
            scheduler = worker.scheduler
            manager = getattr(scheduler, "block_manager", None)
            record = {
                "worker_id": worker.id,
                "running": len(scheduler.running),
                "waiting": len(scheduler.waiting),
                "scheduler_stats": dict(getattr(scheduler, "stats", {})),
            }
            if manager is not None:
                record.update(
                    free_slots=manager.get_num_free_blocks(),
                    used_slots=manager.get_num_allocated_blocks(),
                    dma_pinned_slots=len(getattr(manager, "transfer_references", {})),
                )
            if worker.connector.async_offload:
                record["external_cache"] = worker.connector.describe()
            workers.append(record)
        self.write(
            "cache_windows",
            {
                "simulation_time": env.now,
                "phase": replay.phase,
                "dp_step": (
                    self.engine.dp_coordinator.describe()
                    if getattr(self.engine, "dp_coordinator", None)
                    else None
                ),
                "phase_counts": {p: dict(c) for p, c in replay.phase_counts.items()},
                "workers": workers,
            },
        )
        for file in self.files.values():
            file.flush()

    def monitor(self, env, replay):
        while True:
            self.snapshot(env, replay)
            yield env.timeout(60)

    def close(self):
        for file in self.files.values():
            file.close()
