#!/usr/bin/env python3
"""Probe and run the optional AIC same-shape diagnostic.

TokenSim never imports AIC during formal simulation. This standalone command
records the exact source/runtime identity and returns ``unavailable`` when the
isolated AIC checkout lacks its generated Python/Rust extension. That failure
is explicit rather than silently substituting TokenSim's native tables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def probe(
    aic_root: Path,
    *,
    model: str,
    system: str,
    backend: str,
    systems_path: str | None = None,
    model_path: str | None = None,
    run_estimate: bool = False,
) -> dict:
    source_files = []
    for relative in (
        "src/aiconfigurator/__init__.py",
        "src/aiconfigurator/cli/api.py",
        "aic-core/rust/aiconfigurator-core/Cargo.toml",
    ):
        path = aic_root / relative
        if path.is_file():
            source_files.append({"path": relative, "sha256": _sha(path)})
    result = {
        "schema_version": 1,
        "mode": "aic_same_shape_diagnostic",
        "status": "unavailable",
        "model": model,
        "system": system,
        "backend": backend,
        "aic_root": str(aic_root),
        "source_files": source_files,
        "reason": None,
    }
    sys.path.insert(0, str(aic_root / "src"))
    sys.path.insert(0, str(aic_root / "aic-core" / "src"))
    try:
        import aiconfigurator  # type: ignore
        import aiconfigurator_core._aiconfigurator_core  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external checkout
        result["reason"] = f"AIC import unavailable: {type(exc).__name__}: {exc}"
        result["runtime"] = {
            "python": sys.version,
            "distribution_metadata": (
                importlib.metadata.version("aiconfigurator")
                if any(d.metadata.get("Name") == "aiconfigurator" for d in importlib.metadata.distributions())
                else None
            ),
            "rust_extension": False,
        }
        return result
    result["status"] = "runtime_available"
    result["runtime"] = {
        "python": sys.version,
        "aiconfigurator_version": getattr(aiconfigurator, "__version__", None),
        "rust_extension": True,
    }
    result["reason"] = None
    if run_estimate:
        try:
            from aiconfigurator.cli.api import cli_estimate  # type: ignore

            estimate = cli_estimate(
                model_path or model,
                system,
                backend_name=backend,
                systems_paths=systems_path,
                isl=1024,
                osl=128,
                batch_size=8,
                tp_size=8,
                moe_ep_size=8,
                attention_dp_size=1,
            )
            result["estimate"] = {
                "ttft_ms": _jsonable(estimate.ttft),
                "tpot_ms": _jsonable(estimate.tpot),
                "raw": _jsonable(estimate.raw),
                "per_ops_data": _jsonable(estimate.per_ops_data),
                "per_ops_source": _jsonable(estimate.per_ops_source),
            }
        except Exception as exc:  # preserve per-SKU data gaps
            result["estimate_error"] = f"{type(exc).__name__}: {exc}"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aic-root", type=Path, default=Path("/tmp/aic-full"))
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--system", default="b200_sxm")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--systems-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-estimate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(
        args.aic_root,
        model=args.model,
        system=args.system,
        backend=args.backend,
        systems_path=args.systems_path,
        model_path=args.model_path,
        run_estimate=args.run_estimate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "reason": report["reason"]}, indent=2))
