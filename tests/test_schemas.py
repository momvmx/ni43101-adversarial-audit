import pytest
from pydantic import ValidationError

from ni43101.schemas import ExtractionResult, RawResourceRecord, ResourceRecord


def _raw_record(**changes: object) -> RawResourceRecord:
    values = {
        "document_id": "report-1",
        "location": "Example deposit",
        "category": "Indicated",
        "commodity": "Au",
        "tonnage_value": 8000,
        "tonnage_unit": "kt",
        "grade_value": 3.4,
        "grade_unit": "g/t Au",
        "metal_value": 870,
        "metal_unit": "koz",
        "source_page": 147,
        "table_title": "Mineral Resource Statement",
        "evidence_text": "Indicated 8,000 kt 3.4 g/t Au 870 koz",
        "confidence": 0.95,
    }
    values.update(changes)
    return RawResourceRecord.model_validate(values)


def test_raw_and_extraction_schemas_preserve_source_values() -> None:
    raw = _raw_record()
    result = ExtractionResult(
        document_id="report-1",
        candidate_pages=[147],
        records=[raw],
        uncertain=False,
        uncertainty_reasons=[],
        input_mode="candidate_context",
    )

    assert result.records[0].tonnage_value == 8000
    assert result.records[0].uncertain is False


@pytest.mark.parametrize(
    "category",
    ["Measured", "Measured + Indicated", "Proven", "Probable"],
)
def test_schemas_reject_non_target_categories(category: str) -> None:
    with pytest.raises(ValidationError):
        _raw_record(category=category)


def test_normalized_schema_keeps_original_values() -> None:
    record = ResourceRecord(
        document_id="report-1",
        location="Example deposit",
        category="Inferred",
        commodity="Cu",
        tonnage_mt=100,
        grade=0.5,
        grade_unit="% Cu",
        contained_metal=500_000,
        metal_unit="t",
        source_page=12,
        table_title=None,
        evidence_text="Inferred 100 Mt at 0.5% Cu",
        source_values={"tonnage_value": 100, "tonnage_unit": "Mt"},
        confidence=0.9,
    )

    assert record.source_values["tonnage_unit"] == "Mt"
