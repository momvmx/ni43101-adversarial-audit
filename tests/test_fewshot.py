from datetime import datetime, timezone

from ni43101.evolution import EvolutionRecord, FailureType
from ni43101.fewshot import (
    EvolutionFewShotSelector,
    FewShotQuery,
    query_from_candidate_pages,
)
from ni43101.pdf_locator import CandidatePage


def _record(
    case_id: str,
    document_id: str,
    failure_type: FailureType,
    *,
    commodity: str = "Au",
    tonnage_unit: str = "kt",
    metal_unit: str = "koz",
) -> EvolutionRecord:
    return EvolutionRecord(
        timestamp=datetime(2026, 8, 28, tzinfo=timezone.utc),
        case_id=case_id,
        document_id=document_id,
        pdf_name=f"{document_id}.pdf",
        extractor_model="deepseek-chat",
        critic_model="GLM-4.7-Flash",
        round_number=1,
        candidate_pages=[10],
        commodity=commodity,
        category="Indicated",
        failure_types=[failure_type],
        extractor_output={
            "records": [
                {
                    "tonnage_value": 8_000,
                    "tonnage_unit": tonnage_unit,
                    "grade_value": 3.4,
                    "grade_unit": "g/t Au" if commodity == "Au" else "% Cu",
                    "metal_value": 870,
                    "metal_unit": metal_unit,
                }
            ]
        },
        validator_output=[],
        critic_output={"issues": []},
        decision="evaluation_failed",
        lesson="Ground Truth answer is 999999 and must never be replayed.",
        ground_truth_diff={"truth": 999999},
    )


def _page(text: str) -> CandidatePage:
    return CandidatePage(
        page_index=9,
        page_number=10,
        score=100,
        matched_positive_keywords=[],
        matched_negative_keywords=[],
        reasons=[],
        text=text,
    )


def test_same_document_is_excluded_by_default() -> None:
    selector = EvolutionFewShotSelector(
        records=[
            _record(
                "same",
                "Tanami",
                FailureType.MEASURED_PLUS_INDICATED_CONFUSION,
            ),
            _record(
                "other",
                "OtherMine",
                FailureType.MEASURED_PLUS_INDICATED_CONFUSION,
            ),
        ]
    )
    query = FewShotQuery(
        document_id=" tanami ",
        commodities={"Au"},
        failure_types={FailureType.MEASURED_PLUS_INDICATED_CONFUSION},
        source_units={"kt", "koz"},
        table_structure_risks={"measured_plus_indicated"},
    )

    selected = selector.select(query)

    assert [item.case_id for item in selected] == ["other"]


def test_selects_at_most_three_cases_by_similarity() -> None:
    records = [
        _record(
            f"case-{index}",
            f"doc-{index}",
            FailureType.MEASURED_PLUS_INDICATED_CONFUSION,
        )
        for index in range(5)
    ]
    selector = EvolutionFewShotSelector(records=records)
    query = FewShotQuery(
        document_id="current",
        commodities={"Au"},
        failure_types={FailureType.MEASURED_PLUS_INDICATED_CONFUSION},
        source_units={"kt", "koz"},
        table_structure_risks={"measured_plus_indicated"},
    )

    assert len(selector.select(query)) == 3


def test_prompt_contains_patterns_but_never_ground_truth_answers() -> None:
    selector = EvolutionFewShotSelector(
        records=[
            _record(
                "case-1",
                "OtherMine",
                FailureType.MEASURED_PLUS_INDICATED_CONFUSION,
            )
        ]
    )

    prompt = selector.render(
        "CurrentMine",
        [
            _page(
                "Measured | Indicated | Measured + Indicated | Inferred "
                "Tonnes kt Au g/t contained koz"
            )
        ],
    )

    assert "Previous extraction mistake:" in prompt
    assert "Critic finding:" in prompt
    assert "Lesson:" in prompt
    assert "exact parent header" in prompt
    assert "999999" not in prompt
    assert "8000" not in prompt
    assert "870" not in prompt
    assert "ground_truth_diff" not in prompt


def test_query_detects_commodity_units_and_structure_risks() -> None:
    query = query_from_candidate_pages(
        "report-1",
        [
            _page(
                "Mineral Resource and Mineral Reserve tables; Proven Probable; "
                "Measured + Indicated; Inferred; Au g/t koz; Cu % contained Cu Mt"
            )
        ],
    )

    assert query.commodities == {"Au", "Cu"}
    assert {"g/t", "koz", "mt"} <= query.source_units
    assert FailureType.RESOURCE_RESERVE_CONFUSION in query.failure_types
    assert FailureType.MEASURED_PLUS_INDICATED_CONFUSION in query.failure_types
    assert FailureType.COMMODITY_MISMATCH in query.failure_types
