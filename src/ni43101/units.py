"""Semantic unit conversion for ore tonnage and contained metals."""

from __future__ import annotations

import math

from ni43101.schemas import Commodity


_ORE_TONNAGE_TO_MT = {
    "t": 1 / 1_000_000,
    "kt": 1 / 1_000,
    "mt": 1.0,
}

_GOLD_METAL_TO_OZ = {
    "oz": 1.0,
    "koz": 1_000.0,
    "moz": 1_000_000.0,
}

_COPPER_METAL_TO_T = {
    "t": 1.0,
    "kt": 1_000.0,
    "mt": 1_000_000.0,
}


def _validated_value(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("unit conversion value must be numeric, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError("unit conversion value must be finite and non-negative")
    return converted


def _normalized_unit(unit: str) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must be a non-empty string")
    return unit.strip().casefold()


def _convert(value: float, unit: str, factors: dict[str, float], quantity: str) -> float:
    normalized_unit = _normalized_unit(unit)
    try:
        factor = factors[normalized_unit]
    except KeyError as error:
        supported = ", ".join(factors)
        raise ValueError(
            f"unsupported {quantity} unit {unit!r}; expected one of: {supported}"
        ) from error
    return _validated_value(value) * factor


def normalize_ore_tonnage(value: float, unit: str) -> float:
    """Convert ore tonnage expressed as t, kt, or Mt to Mt of ore."""

    return _convert(value, unit, _ORE_TONNAGE_TO_MT, "ore tonnage")


def normalize_gold_metal(value: float, unit: str) -> float:
    """Convert contained gold expressed as oz, koz, or Moz to oz."""

    return _convert(value, unit, _GOLD_METAL_TO_OZ, "gold metal")


def normalize_copper_metal(value: float, unit: str) -> float:
    """Convert contained copper expressed as t, kt, or Mt to t."""

    return _convert(value, unit, _COPPER_METAL_TO_T, "copper metal")


def normalize_contained_metal(value: float, unit: str, commodity: Commodity) -> float:
    """Normalize contained metal using commodity semantics, not ore semantics."""

    if commodity == "Au":
        return normalize_gold_metal(value, unit)
    if commodity == "Cu":
        return normalize_copper_metal(value, unit)
    raise ValueError(f"unsupported commodity: {commodity!r}")
