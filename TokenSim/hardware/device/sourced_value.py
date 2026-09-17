"""A numeric value annotated with its provenance and evidence grade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import positive_number, require

# Evidence grades — see hardware/README.md for definitions.
#   A = official spec or first-party measurement
#   B = public benchmark (third-party, reproducible)
#   C = calibrated estimate (fitted from measured data)
#   D = project assumption (default, no hardware evidence)
EVIDENCE_GRADES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class SourcedValue:
    """A number together with the evidence that produced it."""

    value: float
    source_id: str
    grade: str = "A"

    @classmethod
    def parse(cls, raw: Any, field_name: str, context: str) -> "SourcedValue":
        if isinstance(raw, Mapping):
            value = positive_number(require(raw, "value", context), field_name, context)
            source_id = str(require(raw, "source_id", context))
            grade = str(raw.get("grade", "A")).upper()
        else:
            raise ConfigurationError(
                f"{context}: {field_name} must be a mapping with value/source_id"
            )
        if grade not in EVIDENCE_GRADES:
            raise ConfigurationError(
                f"{context}: {field_name} grade must be one of {EVIDENCE_GRADES}"
            )
        return cls(value=value, source_id=source_id, grade=grade)

    @classmethod
    def parse_optional_zero(
        cls, raw: Any, field_name: str, context: str, default_source: str
    ) -> "SourcedValue":
        """Like ``parse`` but allows a zero value (used for optional overheads)."""
        if raw is None:
            return cls(value=0.0, source_id=default_source, grade="D")
        if isinstance(raw, Mapping):
            try:
                value = float(require(raw, "value", context))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"{context}: {field_name} must be numeric") from exc
            if value < 0:
                raise ConfigurationError(f"{context}: {field_name} must be non-negative")
            grade = str(raw.get("grade", "D")).upper()
            if grade not in EVIDENCE_GRADES:
                raise ConfigurationError(
                    f"{context}: {field_name} grade must be one of {EVIDENCE_GRADES}"
                )
            return cls(value=value, source_id=str(require(raw, "source_id", context)), grade=grade)
        raise ConfigurationError(f"{context}: {field_name} must be a mapping")
