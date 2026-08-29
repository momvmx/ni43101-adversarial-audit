import pytest

from ni43101.schemas import ResourceRecord
from ni43101.validators import (
    GOLD_GRAMS_PER_TROY_OUNCE,
    validate_category,
    validate_commodity,
    validate_metal_consistency,
)


def test_gold_consistency_calculation() -> None:
    result = validate_metal_consistency(
        commodity="Au",
        tonnage_mt=8,
        grade=3.4,
        actual=870_000,
    )

    assert result.expected == pytest.approx(
        8 * 1_000_000 * 3.4 / GOLD_GRAMS_PER_TROY_OUNCE
    )
    assert result.relative_error < 0.05
    assert result.status == "PASS"


def test_copper_consistency_calculation() -> None:
    result = validate_metal_consistency(
        commodity="Cu",
        tonnage_mt=3930,
        grade=0.43,
        actual=17_000_000,
    )

    assert result.expected == pytest.approx(16_899_000)
    assert result.status == "PASS"


def test_error_above_ten_percent_is_hard_fail() -> None:
    result = validate_metal_consistency(
        commodity="Cu",
        tonnage_mt=100,
        grade=1,
        actual=1_200_000,
    )

    assert result.relative_error == pytest.approx(0.2)
    assert result.status == "HARD_FAIL"


def test_warning_band() -> None:
    result = validate_metal_consistency(
        commodity="Cu",
        tonnage_mt=100,
        grade=1,
        actual=1_075_000,
    )

    assert result.relative_error == pytest.approx(0.075)
    assert result.status == "WARNING"


@pytest.mark.parametrize(
    ("commodity", "grade_unit", "metal_unit"),
    [
        ("Au", "% Cu", "oz"),
        ("Cu", "% Cu", "oz"),
        ("Au", "g/t Au", "t"),
        ("Cu", "g/t Au", "t"),
    ],
)
def test_commodity_mismatch_is_hard_fail(
    commodity: str,
    grade_unit: str,
    metal_unit: str,
) -> None:
    assert validate_commodity(commodity, grade_unit, metal_unit).status == "HARD_FAIL"


@pytest.mark.parametrize(
    "category",
    ["Measured", "Measured + Indicated", "Proven", "Probable"],
)
def test_category_validator_rejects_non_targets(category: str) -> None:
    assert validate_category(category).status == "HARD_FAIL"


def test_record_with_commodity_mismatch_is_hard_fail() -> None:
    record = ResourceRecord(
        document_id="report-1",
        location="Example",
        category="Indicated",
        commodity="Au",
        tonnage_mt=8,
        grade=3.4,
        grade_unit="% Cu",
        contained_metal=870_000,
        metal_unit="oz",
        source_page=1,
        table_title=None,
        evidence_text="source row",
        source_values={},
        confidence=0.8,
    )

    assert validate_metal_consistency(record).status == "HARD_FAIL"
