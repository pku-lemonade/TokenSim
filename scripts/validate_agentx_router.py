#!/usr/bin/env python3
"""Validate the router against all historical key-to-worker log mappings."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TokenSim.placement.consistent_hash import ConsistentHashRouter, fbi_hash


def audit(log_root, source_root):
    router = ConsistentHashRouter([SimpleNamespace(id=i, dp_rank=i) for i in range(8)])
    pattern = re.compile(
        r"CONSISTENT_HASH_DEBUG: Hash key '(.*?)' mapped to worker: (\S+)@(\d+)"
    )
    points, gold = {}, {}
    for sku in ("b200", "b300"):
        path = log_root / sku / "agentic/router.log"
        matches = [
            m.groups() for m in pattern.finditer(path.read_text(errors="replace"))
        ]
        if not matches:
            raise ValueError(f"no observable router mappings in {path}")
        wrong = [
            (key, int(rank), router.rank_for_key(key))
            for key, url, rank in matches
            if router.rank_for_key(key) != int(rank)
        ]
        points[sku] = {
            "requests": len(matches),
            "sessions": len({m[0] for m in matches}),
            "mismatches": len(wrong),
            "examples": wrong[:20],
            "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, _, rank in matches:
            if int(rank) not in gold:
                gold[int(rank)] = {
                    "key": key,
                    "hash": fbi_hash(key),
                    "expected_rank": int(rank),
                }
    files = (
        "src/policies/consistent_hash.rs",
        "src/policies/hash_key.rs",
        "src/routers/http/dp_utils.rs",
        "LICENSE",
    )
    return {
        "source": {
            "repository": "https://github.com/vllm-project/router",
            "tag": "v0.1.14",
            "revision": "0a6f74ec88c87b5477dac8e24466ab760b9f1f35",
            "files": {
                f: hashlib.sha256((source_root / f).read_bytes()).hexdigest()
                for f in files
            },
        },
        "backend_url": router.backend_url,
        "virtual_nodes_per_worker": 160,
        "hash_key_header": "x-session-id, aliased from benchmark correlation ID",
        "checkpoints": points,
        "observed_vectors": list(gold.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("/tmp/infx-art"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/tmp/agentx-router-reference/router-0.1.14"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/reference/agentx-router/validation.json",
    )
    args = parser.parse_args()
    result = audit(args.log_root, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checkpoints"], indent=2))
    if any(p["mismatches"] for p in result["checkpoints"].values()):
        raise SystemExit(1)
