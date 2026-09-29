"""逐条散列完整服务记录，避免同时持有两个大型 profiling 文件。"""

import argparse
import gc
import hashlib
import json
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, case in data["cases"].items():
        services = case.pop("services")
        digest = hashlib.sha256()
        for row in services:
            digest.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            )
        case["services_sha256"] = digest.hexdigest()
        result[name] = case
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = summarize(args.reference)
    gc.collect()
    optimized = summarize(args.optimized)
    assert reference.keys() == optimized.keys(), "profiling cases differ"
    for name in reference:
        first, second = reference[name], optimized[name]
        for key in ("services_sha256", "service_count", "communication_bytes"):
            assert first[key] == second[key], (name, key)
        simulation = lambda case: {
            k: v
            for k, v in case["result"].items()
            if k not in {"operator_stats", "optimization_stats"}
        }
        assert simulation(first) == simulation(second), (name, "simulation result")
        print(
            f"{name}: exact match; wall {first['wall_seconds']:.3f}s -> {second['wall_seconds']:.3f}s"
        )
    args.output.write_text(
        json.dumps(
            dict(reference=reference, optimized=optimized), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
