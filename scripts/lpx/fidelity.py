"""Load auditable LP30 hypotheses without changing global device defaults."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from TokenSim.af.runtime.lpx_memory import LP30Memory
from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import DeviceCatalog
from TokenSim.operator_data.analytical.lp30 import LP30Pipeline
from scripts.lpx.fabric import LP30Fabric
from scripts.lpx.token_units import TokenUnits

SECTIONS = {
    "operator": LP30Pipeline,
    "memory": LP30Memory,
    "fabric": LP30Fabric,
    "token_units": TokenUnits,
}


def validate(raw):
    if not isinstance(raw, dict) or set(raw) - (
        set(SECTIONS) | {"compute", "description"}
    ):
        raise ConfigurationError("unknown LP30 fidelity configuration fields")
    result = {}
    try:
        for name, cls in SECTIONS.items():
            if name in raw:
                if not isinstance(raw[name], dict):
                    raise ConfigurationError(f"LP30 {name} requires an object")
                result[name] = asdict(cls(**raw[name]))
        compute = raw.get("compute", {})
        if not isinstance(compute, dict) or set(compute) - {
            "gemm_mfu",
            "attention_mfu",
            "memory_efficiency",
        }:
            raise ConfigurationError("unknown LP30 compute assumptions")
        for name, value in compute.items():
            number(value, f"LP30 {name}", positive=True, maximum=1)
        result["compute"] = dict(compute)
    except TypeError as exc:
        raise ConfigurationError(f"invalid LP30 fidelity configuration: {exc}") from exc
    if (
        "operator" in result
        and "memory" in result
        and result["operator"]["attention_tile_keys"]
        != result["memory"]["attention_tile_keys"]
    ):
        raise ConfigurationError("LP30 operator and memory attention tiles must agree")
    if "description" in raw:
        result["description"] = str(raw["description"])
    return result


def load(path):
    try:
        return validate(json.loads(Path(path).read_text()))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"cannot load LP30 fidelity configuration: {exc}"
        ) from exc


def with_fidelity(hardware, raw):
    raw = validate(raw)
    parameters = dict(raw.get("compute", {}))
    for section, key in (
        ("operator", "lpx_pipeline"),
        ("memory", "lpx_memory"),
        ("fabric", "lpx_fabric"),
    ):
        if section in raw:
            parameters[key] = raw[section]
    return replace(
        hardware,
        devices=DeviceCatalog(
            (
                replace(d, analytical={**d.analytical, **parameters})
                if d.device_id == "groq_lp30"
                else d
            )
            for d in hardware.devices
        ),
        _packages={},
    )
