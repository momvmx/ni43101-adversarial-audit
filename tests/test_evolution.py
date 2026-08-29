from datetime import datetime, timezone

import orjson
import pytest

from ni43101.evolution import (
    EvolutionLogger,
    EvolutionRecord,
    FailureType,
    lesson_for_failure_types,
)


def _record(case_id: str, failure_type: FailureType) -> EvolutionRecord:
    return EvolutionRecord(
        timestamp=datetime(2026, 8, 28, tzinfo=timezone.utc),
        case_id=case_id,
        document_id="report-1",
        pdf_name="report-1.pdf",
        extractor_model="deepseek-chat",
        critic_model="GLM-4.7-Flash",
        round_number=1,
        candidate_pages=[10, 11],
        commodity="Au",
        category="Indicated",
        failure_types=[failure_type],
        extractor_output={"records": []},
        validator_output=[],
        critic_output={"score": 6},
        decision="revise",
        lesson=lesson_for_failure_types([failure_type]),
    )


def test_append_twice_preserves_both_jsonl_records(tmp_path) -> None:
    path = tmp_path / "evolution.jsonl"
    logger = EvolutionLogger(path)

    assert logger.append(_record("case-1", FailureType.LOW_CRITIC_SCORE))
    assert logger.append(_record("case-2", FailureType.METAL_UNIT_ERROR))

    lines = path.read_bytes().splitlines()
    assert len(lines) == 2
    assert orjson.loads(lines[0])["case_id"] == "case-1"
    assert orjson.loads(lines[1])["case_id"] == "case-2"
    assert [record.case_id for record in logger.read()] == ["case-1", "case-2"]


def test_bad_json_line_is_skipped_without_breaking_read(tmp_path) -> None:
    path = tmp_path / "evolution.jsonl"
    logger = EvolutionLogger(path)
    logger.append(_record("valid-case", FailureType.MISSING_EVIDENCE))
    with path.open("ab") as stream:
        stream.write(b"{bad json}\n")

    with pytest.warns(RuntimeWarning, match="skipping invalid evolution log line"):
        records = logger.read()

    assert [record.case_id for record in records] == ["valid-case"]


def test_write_failure_warns_and_does_not_raise(tmp_path) -> None:
    logger = EvolutionLogger(tmp_path)

    with pytest.warns(RuntimeWarning, match="append failed"):
        written = logger.append(_record("case-1", FailureType.AMBIGUOUS_TABLE))

    assert written is False


def test_required_lessons_are_generated_from_failure_type() -> None:
    lesson = lesson_for_failure_types(
        [FailureType.MEASURED_PLUS_INDICATED_CONFUSION]
    )

    assert "exact parent header" in lesson
    assert "Measured + Indicated" in lesson


def test_evaluation_failure_entry_appends_ground_truth_diff(tmp_path) -> None:
    logger = EvolutionLogger(tmp_path / "evolution.jsonl")

    assert logger.log_evaluation_failure(
        case_id="eval-case",
        document_id="report-1",
        pdf_name="report-1.pdf",
        extractor_model="deepseek-chat",
        critic_model="GLM-4.7-Flash",
        round_number=3,
        candidate_pages=[10],
        failure_types=[FailureType.MISSING_CATEGORY],
        ground_truth_diff={"missing": ["Inferred"]},
    )

    record = logger.read()[0]
    assert record.decision == "evaluation_failed"
    assert record.ground_truth_diff == {"missing": ["Inferred"]}
