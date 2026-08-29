import orjson
import pytest

from ni43101.config import Settings
from ni43101.critic import (
    CriticConfigurationError,
    CriticMaster,
    FakeCriticClient,
)
from ni43101.pdf_locator import CandidatePage
from ni43101.schemas import ExtractionResult, RawResourceRecord, ResourceRecord
from ni43101.validators import (
    DeterministicValidationResult,
    MetalConsistencyResult,
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


def _raw_record() -> RawResourceRecord:
    return RawResourceRecord(
        document_id="report-1",
        location="Example deposit",
        category="Indicated",
        commodity="Au",
        tonnage_value=8000,
        tonnage_unit="kt",
        grade_value=3.4,
        grade_unit="g/t Au",
        metal_value=870,
        metal_unit="koz",
        source_page=10,
        table_title="Mineral Resource Statement",
        evidence_text="Indicated 8,000 kt 3.40 g/t Au 870 koz",
        confidence=0.95,
    )


def _raw_extraction() -> ExtractionResult:
    return ExtractionResult(
        document_id="report-1",
        candidate_pages=[10],
        records=[_raw_record()],
        uncertain=False,
        uncertainty_reasons=[],
        input_mode="candidate_pages_with_context",
    )


def _normalized_record() -> ResourceRecord:
    return ResourceRecord(
        document_id="report-1",
        location="Example deposit",
        category="Indicated",
        commodity="Au",
        tonnage_mt=8,
        grade=3.4,
        grade_unit="g/t Au",
        contained_metal=870_000,
        metal_unit="oz",
        source_page=10,
        table_title="Mineral Resource Statement",
        evidence_text="Indicated 8,000 kt 3.40 g/t Au 870 koz",
        source_values=_raw_record().model_dump(),
        confidence=0.95,
    )


def _validation(
    *,
    relative_error: float = 0.005,
    status: str = "PASS",
) -> MetalConsistencyResult:
    return MetalConsistencyResult.model_validate(
        {
            "expected": 874_500,
            "actual": 870_000,
            "relative_error": relative_error,
            "status": status,
        }
    )


def _checks(**changes: str) -> dict[str, str]:
    checks = {
        "resource_vs_reserve": "pass",
        "category_alignment": "pass",
        "row_alignment": "pass",
        "commodity_alignment": "pass",
        "unit_correctness": "pass",
        "math_consistency": "pass",
        "evidence_support": "pass",
        "completeness": "pass",
    }
    checks.update(changes)
    return checks


def _critic_result(
    *,
    score: int,
    verdict: str,
    issues: list[dict[str, object]] | None = None,
    checks: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "score": score,
        "verdict": verdict,
        "issues": issues or [],
        "checks": checks or _checks(),
        "summary": "Adversarial review completed.",
    }


def _critical_issue(code: str, message: str) -> dict[str, object]:
    return {
        "code": code,
        "severity": "critical",
        "record_index": 0,
        "field": "category",
        "message": message,
        "source_evidence": "Source table header and row text",
        "suggested_action": "Revise the extraction using the correct column",
    }


def _review(
    fake_result: dict[str, object],
    *,
    source_text: str = "Mineral Resource Statement Indicated 8,000 kt 3.40 870",
    validation: MetalConsistencyResult | None = None,
):
    fake = FakeCriticClient([fake_result])
    result = CriticMaster(fake).review(
        [_page(source_text)],
        _raw_extraction(),
        [_normalized_record()],
        [validation or _validation()],
    )
    return result, fake


def test_correct_data_keeps_score_nine_and_passes() -> None:
    result, _ = _review(_critic_result(score=9, verdict="pass"))

    assert result.score == 9
    assert result.verdict == "pass"
    assert result.issues == []


def test_measured_plus_indicated_confusion_is_critical() -> None:
    issue = _critical_issue(
        "measured_plus_indicated_as_indicated",
        "Measured + Indicated was treated as Indicated",
    )
    result, _ = _review(
        _critic_result(
            score=4,
            verdict="revise",
            issues=[issue],
            checks=_checks(category_alignment="critical"),
        ),
        source_text=(
            "Measured | Indicated | Measured + Indicated | Inferred\n"
            "Measured + Indicated 8,000 3.40 870"
        ),
    )

    assert result.checks.category_alignment == "critical"
    assert result.issues[0].severity == "critical"
    assert result.score <= 7


def test_reserve_probable_confusion_is_critical() -> None:
    issue = _critical_issue(
        "reserve_as_resource",
        "A Probable Mineral Reserve row was extracted as a Resource",
    )
    result, _ = _review(
        _critic_result(
            score=2,
            verdict="abstain",
            issues=[issue],
            checks=_checks(resource_vs_reserve="critical"),
        ),
        source_text="Mineral Reserve Statement Proven Probable 8,000 3.40 870",
    )

    assert result.checks.resource_vs_reserve == "critical"
    assert result.issues[0].severity == "critical"
    assert result.verdict == "abstain"


def test_twenty_percent_math_error_caps_fake_score_nine_at_seven() -> None:
    result, _ = _review(
        _critic_result(score=9, verdict="pass"),
        validation=_validation(relative_error=0.20, status="HARD_FAIL"),
    )

    assert result.score == 7
    assert result.verdict == "revise"
    assert result.checks.math_consistency == "critical"
    assert any(
        issue.code == "math_consistency_hard_fail"
        and issue.severity == "critical"
        for issue in result.issues
    )


def test_relative_error_overrides_incorrect_upstream_pass_status() -> None:
    result, _ = _review(
        _critic_result(score=10, verdict="pass"),
        validation=_validation(relative_error=0.20, status="PASS"),
    )

    assert result.score == 7
    assert result.verdict == "revise"
    assert result.checks.math_consistency == "critical"


def test_warning_math_result_caps_ten_at_nine() -> None:
    result, _ = _review(
        _critic_result(score=10, verdict="pass"),
        validation=_validation(relative_error=0.075, status="WARNING"),
    )

    assert result.score == 9
    assert result.verdict == "pass"
    assert result.checks.math_consistency == "warning"


def test_pipeline_validation_wrapper_supports_failed_normalization() -> None:
    fake = FakeCriticClient([_critic_result(score=9, verdict="pass")])
    result = CriticMaster(fake).review(
        [_page("Mineral Resource Statement Indicated")],
        _raw_extraction(),
        [],
        [
            DeterministicValidationResult(
                record_index=0,
                normalized_record_index=None,
                status="HARD_FAIL",
                messages=["unsupported contained copper unit"],
            )
        ],
    )

    assert result.score == 7
    assert result.verdict == "revise"
    assert any(issue.code == "deterministic_hard_fail" for issue in result.issues)


def test_critic_payload_contains_only_allowed_evidence() -> None:
    _, fake = _review(_critic_result(score=9, verdict="pass"))

    payload = orjson.loads(fake.calls[0]["messages"][1]["content"])
    assert set(payload) == {
        "task",
        "candidate_pages",
        "deepseek_raw_extraction",
        "normalized_records",
        "deterministic_validator_results",
        "output_schema",
    }
    assert "ground_truth" not in payload


def test_extractor_and_critic_models_must_differ() -> None:
    settings = Settings(
        extractor_api_key="extractor-test",
        extractor_model="same-model",
        critic_api_key="critic-test",
        critic_base_url="https://critic.invalid",
        critic_model="same-model",
    )

    with pytest.raises(CriticConfigurationError, match="different models"):
        CriticMaster.from_settings(settings)
