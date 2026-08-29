"""Deterministic validators for normalized Mineral Resource records."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ni43101.schemas import Commodity, ResourceRecord


ValidationStatus = Literal["PASS", "WARNING", "HARD_FAIL"]

GOLD_GRAMS_PER_TROY_OUNCE = 31.1034768
PASS_RELATIVE_ERROR = 0.05
WARNING_RELATIVE_ERROR = 0.10

_COMMODITY_UNITS: dict[str, tuple[str, str]] = {
    "Au": ("g/t Au", "oz"),
    "Cu": ("% Cu", "t"),
}

_ALLOWED_CATEGORIES = {"Indicated", "Inferred"}


class MetalConsistencyResult(BaseModel):
    """Comparison between reported and grade-derived contained metal."""

    model_config = ConfigDict(extra="forbid")

    expected: float
    actual: float
    relative_error: float
    status: ValidationStatus


class DeterministicValidationResult(BaseModel):
    """Per-raw-record validation evidence used by the orchestration pipeline."""

    model_config = ConfigDict(extra="forbid")

    record_index: int = Field(ge=0)
    normalized_record_index: int | None = Field(default=None, ge=0)
    expected: float | None = None
    actual: float | None = None
    relative_error: float | None = Field(default=None, ge=0)
    status: ValidationStatus
    messages: list[str]


class RuleValidationResult(BaseModel):
    """Result for a deterministic categorical rule."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["PASS", "HARD_FAIL"]
    reason: str | None = None


def validate_commodity(
    commodity: str,
    grade_unit: str,
    metal_unit: str,
) -> RuleValidationResult:
    """Validate the commodity, grade-unit, and metal-unit combination."""

    expected_units = _COMMODITY_UNITS.get(commodity)
    if expected_units is None:
        return RuleValidationResult(
            status="HARD_FAIL",
            reason=f"unsupported commodity: {commodity!r}",
        )

    expected_grade_unit, expected_metal_unit = expected_units
    if grade_unit != expected_grade_unit or metal_unit != expected_metal_unit:
        return RuleValidationResult(
            status="HARD_FAIL",
            reason=(
                f"{commodity} requires grade unit {expected_grade_unit!r} and "
                f"metal unit {expected_metal_unit!r}; received "
                f"{grade_unit!r} and {metal_unit!r}"
            ),
        )

    return RuleValidationResult(status="PASS")


def validate_commodity_units(
    commodity: str,
    grade_unit: str,
    metal_unit: str,
) -> RuleValidationResult:
    """Explicitly named alias for :func:`validate_commodity`."""

    return validate_commodity(commodity, grade_unit, metal_unit)


def validate_category(category: str) -> RuleValidationResult:
    """Allow only final Indicated and Inferred records."""

    if category in _ALLOWED_CATEGORIES:
        return RuleValidationResult(status="PASS")
    return RuleValidationResult(
        status="HARD_FAIL",
        reason=(
            f"category {category!r} is not a target Mineral Resource category; "
            "only 'Indicated' and 'Inferred' are allowed"
        ),
    )


def _expected_metal(commodity: Commodity, tonnage_mt: float, grade: float) -> float:
    if commodity == "Au":
        return tonnage_mt * 1_000_000 * grade / GOLD_GRAMS_PER_TROY_OUNCE
    if commodity == "Cu":
        return tonnage_mt * 1_000_000 * grade / 100
    raise ValueError(f"unsupported commodity: {commodity!r}")


def _finite_non_negative(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return converted


def validate_metal_consistency(
    record: ResourceRecord | None = None,
    *,
    commodity: Commodity | None = None,
    tonnage_mt: float | None = None,
    grade: float | None = None,
    actual: float | None = None,
) -> MetalConsistencyResult:
    """Validate reported metal against tonnage and grade.

    Pass either a normalized ``ResourceRecord`` or all four keyword values.
    A record with a commodity/unit mismatch is always a hard failure.
    """

    commodity_validation: RuleValidationResult | None = None
    if record is not None:
        if any(value is not None for value in (commodity, tonnage_mt, grade, actual)):
            raise ValueError("pass either record or explicit values, not both")
        commodity = record.commodity
        tonnage_mt = record.tonnage_mt
        grade = record.grade
        actual = record.contained_metal
        commodity_validation = validate_commodity(
            record.commodity,
            record.grade_unit,
            record.metal_unit,
        )

    if commodity is None or tonnage_mt is None or grade is None or actual is None:
        raise ValueError(
            "commodity, tonnage_mt, grade, and actual are required without a record"
        )

    normalized_tonnage = _finite_non_negative(tonnage_mt, "tonnage_mt")
    normalized_grade = _finite_non_negative(grade, "grade")
    normalized_actual = _finite_non_negative(actual, "actual")
    expected = _expected_metal(commodity, normalized_tonnage, normalized_grade)

    if expected == 0:
        relative_error = 0.0 if normalized_actual == 0 else math.inf
    else:
        relative_error = abs(normalized_actual - expected) / expected

    if commodity_validation is not None and commodity_validation.status == "HARD_FAIL":
        status: ValidationStatus = "HARD_FAIL"
    elif relative_error <= PASS_RELATIVE_ERROR:
        status = "PASS"
    elif relative_error <= WARNING_RELATIVE_ERROR:
        status = "WARNING"
    else:
        status = "HARD_FAIL"

    return MetalConsistencyResult(
        expected=expected,
        actual=normalized_actual,
        relative_error=relative_error,
        status=status,
    )
