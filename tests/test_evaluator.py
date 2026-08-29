import orjson
import pytest
from pydantic import ValidationError

from ni43101.evaluator import (
    Evaluator,
    GroundTruth,
    GroundTruthRecord,
    aggregate_metrics,
    evaluate_document,
    field_is_correct,
    normalize_location,
    write_evaluation_report,
)
from ni43101.evolution import EvolutionLogger
from ni43101.orchestrator import PipelineResult
from ni43101.schemas import RawResourceRecord, ResourceRecord


def _truth() -> GroundTruth:
    return GroundTruth(
        document_id="report-1",
        records=[
            GroundTruthRecord(
                location="Tanami UG",
                category="Indicated",
                commodity="Au",
                tonnage_mt=100,
                grade=3.4,
                grade_unit="g/t Au",
                contained_metal=10_000_000,
                metal_unit="oz",
            )
        ],
    )


def _normalized_record(tonnage_mt: float) -> ResourceRecord:
    return ResourceRecord(
        document_id="report-1",
        location="  TANAMI---UG ",
        category="Indicated",
        commodity="Au",
        tonnage_mt=tonnage_mt,
        grade=3.4,
        grade_unit="g/t Au",
        contained_metal=10_000_000,
        metal_unit="oz",
        source_page=10,
        table_title="Mineral Resource Statement",
        evidence_text="Indicated source row",
        source_values={},
        confidence=0.95,
    )


def _raw_candidate() -> RawResourceRecord:
    return RawResourceRecord(
        document_id="report-1",
        location="Tanami UG",
        category="Indicated",
        commodity="Au",
        tonnage_value=105.1,
        tonnage_unit="Mt",
        grade_value=3.4,
        grade_unit="g/t Au",
        metal_value=10,
        metal_unit="Moz",
        source_page=10,
        table_title="Mineral Resource Statement",
        evidence_text="Indicated source row",
        confidence=0.5,
    )


def _pipeline(
    status: str,
    *,
    tonnage_mt: float = 100,
) -> PipelineResult:
    return PipelineResult.model_validate(
        {
            "document_id": "report-1",
            "status": status,
            "needs_human_review": status == "abstain",
            "accepted_records": (
                [_normalized_record(tonnage_mt).model_dump()] if status == "pass" else []
            ),
            "candidate_records": (
                [_raw_candidate().model_dump()] if status == "abstain" else []
            ),
            "rounds": [],
            "final_score": 9,
            "abstain_reason": "human review required" if status == "abstain" else None,
        }
    )


def test_five_percent_numeric_boundary_is_inclusive() -> None:
    assert field_is_correct(104, 100)
    assert field_is_correct(105, 100)
    assert not field_is_correct(105.1, 100)


def test_wrong_data_with_pass_is_unsafe_accept() -> None:
    detail = evaluate_document(_pipeline("pass", tonnage_mt=105.1), _truth())
    metrics = aggregate_metrics([detail])

    assert detail.unsafe_accept is True
    assert metrics.documents_passed == 1
    assert metrics.unsafe_accept_count == 1
    assert metrics.unsafe_accept_rate == 1
    assert metrics.correct_numeric_fields == 2
    assert metrics.total_numeric_fields == 3


def test_wrong_candidate_with_abstain_is_not_unsafe_accept() -> None:
    detail = evaluate_document(_pipeline("abstain"), _truth())
    metrics = aggregate_metrics([detail])

    assert detail.unsafe_accept is False
    assert detail.safe_abstain is True
    assert metrics.documents_abstained == 1
    assert metrics.safe_abstain_count == 1
    assert metrics.unsafe_accept_count == 0
    assert metrics.unsafe_accept_rate == 0
    assert metrics.correct_numeric_fields == 0


def test_location_is_normalized_but_category_is_never_fuzzy() -> None:
    assert normalize_location(" Tanami---UG ") == normalize_location("tanami ug")
    with pytest.raises(ValidationError):
        GroundTruthRecord.model_validate(
            {
                **_truth().records[0].model_dump(),
                "category": "Measured + Indicated",
            }
        )


def test_evaluator_logs_failed_evaluation_and_writes_report(tmp_path) -> None:
    evolution_path = tmp_path / "evolution.jsonl"
    evaluator = Evaluator(evolution_logger=EvolutionLogger(evolution_path))

    metrics, details = evaluator.evaluate(
        [_pipeline("pass", tonnage_mt=105.1)],
        {"report-1": _truth()},
    )
    report_path = write_evaluation_report(
        metrics,
        tmp_path / "outputs" / "evaluation_report.json",
    )

    assert details[0].unsafe_accept
    assert len(evolution_path.read_bytes().splitlines()) == 1
    evolution_event = orjson.loads(evolution_path.read_bytes().splitlines()[0])
    assert evolution_event["decision"] == "evaluation_failed"
    assert evolution_event["ground_truth_diff"]["numeric_errors"]
    report = orjson.loads(report_path.read_bytes())
    assert report["unsafe_accept_count"] == 1
    assert report["field_accuracy"] == pytest.approx(2 / 3)
