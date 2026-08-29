import pytest

from ni43101.units import (
    normalize_contained_metal,
    normalize_copper_metal,
    normalize_gold_metal,
    normalize_ore_tonnage,
)


def test_ore_tonnage_8000_kt_to_8_mt() -> None:
    assert normalize_ore_tonnage(8000, "kt") == pytest.approx(8)


def test_gold_870_koz_to_870000_oz() -> None:
    assert normalize_gold_metal(870, "koz") == pytest.approx(870_000)


def test_gold_1_37_moz_to_1370000_oz() -> None:
    assert normalize_gold_metal(1.37, "Moz") == pytest.approx(1_370_000)


def test_copper_17_mt_to_17000000_t() -> None:
    assert normalize_copper_metal(17, "Mt") == pytest.approx(17_000_000)


def test_mt_is_interpreted_by_quantity_semantics() -> None:
    assert normalize_ore_tonnage(17, "Mt") == pytest.approx(17)
    assert normalize_contained_metal(17, "Mt", "Cu") == pytest.approx(17_000_000)


def test_gold_rejects_copper_units() -> None:
    with pytest.raises(ValueError, match="gold metal"):
        normalize_contained_metal(17, "Mt", "Au")
