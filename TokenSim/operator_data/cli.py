from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from TokenSim.operator_data.analytical import Calibration, HardwareProfile
from TokenSim.operator_data.generator import (
    CalibrationRule,
    EvidenceGrade,
    Observation,
    OperatorDataGenerator,
    SourceRecord,
)
from TokenSim.operator_data.package import OperatorDataPackage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or validate operator-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--input", required=True)
    generate.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("package")
    args = parser.parse_args(argv)
    if args.command == "validate":
        package = OperatorDataPackage.load(args.package)
        print(
            f"valid operator-data package: dataset={package.dataset_version}, "
            f"schema={package.schema_version}"
        )
        return 0

    document = _load_yaml(Path(args.input))
    profiles = {
        name: HardwareProfile(device=name, **values)
        for name, values in document["hardware_profiles"].items()
    }
    sources = [
        SourceRecord(
            **{
                **source,
                "evidence_grade": EvidenceGrade(source["evidence_grade"]),
            }
        )
        for source in document["sources"]
    ]
    observations = [
        Observation(
            table_name=item["table"],
            key=item["key"],
            latency_us=item["latency_us"],
            source_id=item["source_id"],
            split=item.get("split", "runtime"),
        )
        for item in document["observations"]
    ]
    calibrations = {
        name: Calibration(**values)
        for name, values in document.get("calibrations", {}).items()
    }
    generator = OperatorDataGenerator(
        dataset_version=document["dataset_version"],
        hardware_profiles=profiles,
        sources=sources,
        calibrations=calibrations,
        shape_calibrations=[CalibrationRule(
            table_name=item["table_name"], key_match=item["key_match"],
            calibration=Calibration(**item["calibration"]))
            for item in document.get("shape_calibrations", [])],
    )
    generator.generate(
        args.output,
        experiment_id=document["experiment_id"],
        shape_requests=document["shape_requests"],
        observations=observations,
        holdout_comparisons=document.get("holdout_comparisons"),
    )
    package = OperatorDataPackage.load(args.output)
    print(f"generated operator-data package: {package.root}")
    return 0


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("operator-data CLI requires PyYAML") from exc
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("generator input root must be a mapping")
    return document


if __name__ == "__main__":
    raise SystemExit(main())
