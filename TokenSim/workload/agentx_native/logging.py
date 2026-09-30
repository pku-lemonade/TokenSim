"""Streaming evidence for native replay and physical cache pressure."""

from __future__ import annotations

import json
from pathlib import Path


class NativeReplayLogs:
    def __init__(self, directory, engine):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.files = {
            name: (directory / f"{name}.jsonl").open("w")
            for name in ("events", "requests", "cache_windows")
        }
        (directory / "cache_configuration.json").write_text(
            json.dumps(
                [
                    {"worker_id": w.id, "cache": w.cache_config.to_dict()}
                    for w in engine.workers
                ],
                indent=2,
            )
            + "\n"
        )

    def write(self, name, record):
        self.files[name].write(json.dumps(record, separators=(",", ":")) + "\n")

    def snapshot(self, env, replay):
        workers = []
        for worker in self.engine.workers:
            manager = worker.scheduler.block_manager
            workers.append(
                {
                    "worker_id": worker.id,
                    "running": len(worker.scheduler.running),
                    "waiting": len(worker.scheduler.waiting),
                    "free_slots": manager.get_num_free_blocks(),
                    "watermark_slots": manager.watermark_blocks,
                    "used_slots": manager.get_num_allocated_blocks(),
                    "peak_used_slots": manager.peak_used_slots,
                    "allocations": manager.allocation_count,
                    "releases": manager.release_count,
                    "evictions": manager.eviction_count,
                    "cached_slots": len(manager.cache)
                    + sum(len(copies) for copies in manager.duplicates.values()),
                }
            )
        self.write(
            "cache_windows",
            {
                "simulation_time": env.now,
                "phase": replay.phase,
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
