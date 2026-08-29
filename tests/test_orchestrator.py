from collections.abc import Sequence
from pathlib import Path

import orjson
import pytest

from ni43101.critic import (
    CriticChecks,
    CriticIssue,
    CriticOutputError,
    CriticResult,
)
from ni43101.extractor import ExtractorOutputError
from ni43101.orchestrator import NI43101Orchestrator
from ni43101.pdf_locator import CandidatePage
from ni43101.schemas import ExtractionResult, RawResourceRecord


@pytest.fixture(autouse=True)
def _isolate_evolution_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)


def _page() -> CandidatePage:
    return CandidatePage(
        page_index=9,
        page_number=10,
        score=100,
        matched_positive_keywords=["Mineral Resource Statement", "Indicated"],
        matched_negative_keywords=[],
        reasons=["strong resource title"],
        text=(
            "Mineral Resource Statement\n"
            "Indicated 8,000 kt 3.40 g/t Au 870 koz"
        ),
    )


def _raw_record(
    *,
    metal_value: float = 870,
    commodity: str = "Au",
    metal_unit: str = "koz",
    grade_value: float = 3.4,
    grade_unit: str = "g/t Au",
    tonnage_value: float = 8000,
    tonnage_unit: str = "kt",
) -> RawResourceRecord:
    return RawResourceRecord.model_validate(
        {
            "document_id": "report-1",
            "location": "Example deposit",
            "category": "Indicated",
            "commodity": commodity,
            "tonnage_value": tonnage_value,
            "tonnage_unit": tonnage_unit,
            "grade_value": grade_value,
            "grade_unit": grade_unit,
            "metal_value": metal_value,
            "metal_unit": metal_unit,
            "source_page": 10,
            "table_title": "Mineral Resource Statement",
            "evidence_text": "Indicated source table row",
            "confidence": 0.95,
        }
    )


def _extraction(
    record: RawResourceRecord | None = None,
    *,
    uncertain: bool = False,
) -> ExtractionResult:
    return ExtractionResult(
        document_id="report-1",
        candidate_pages=[10],
        records=[record or _raw_record()],
        uncertain=uncertain,
        uncertainty_reasons=["column alignment is ambiguous"] if uncertain else [],
        input_mode="candidate_pages_with_context",
    )


def _checks() -> CriticChecks:
    return CriticChecks(
        resource_vs_reserve="pass",
        category_alignment="pass",
        row_alignment="pass",
        commodity_alignment="pass",
        unit_correctness="pass",
        math_consistency="pass",
        evidence_support="pass",
        completeness="pass",
    )


def _critic_result(score: int, verdict: str = "pass") -> CriticResult:
    return CriticResult.model_validate(
        {
            "score": score,
            "verdict": verdict,
            "issues": [],
            "checks": _checks().model_dump(),
            "summary": "Audit completed.",
        }
    )


class FakeExtractor:
    def __init__(self, results: Sequence[ExtractionResult]) -> None:
        self.results = list(results)
        self.extract_calls = 0
        self.revision_calls: list[dict[str, object]] = []

    def extract(self, *args: object, **kwargs: object) -> ExtractionResult:
        self.extract_calls += 1
        return self.results.pop(0)

    def revise(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage] | None,
        previous_extraction: ExtractionResult,
        validator_failures: Sequence[object],
        critic_issues: Sequence[CriticIssue],
        **kwargs: object,
    ) -> ExtractionResult:
        self.revision_calls.append(
            {
                "document_id": document_id,
                "previous_extraction": previous_extraction,
                "validator_failures": list(validator_failures),
                "critic_issues": list(critic_issues),
            }
        )
        return self.results.pop(0)


class FakeCritic:
    def __init__(self, results: Sequence[CriticResult]) -> None:
        self.results = list(results)
        self.calls = 0

    def review(self, *args: object, **kwargs: object) -> CriticResult:
        self.calls += 1
        return self.results.pop(0)


def test_correct_extraction_score_nine_and_validator_passes() -> None:
    extractor = FakeExtractor([_extraction()])
    critic = FakeCritic([_critic_result(9)])

    result = NI43101Orchestrator(extractor, critic).run_with_pages(
        "report-1",
        [_page()],
    )

    assert result.status == "pass"
    assert result.needs_human_review is False
    assert len(result.accepted_records) == 1
    assert result.accepted_records[0].tonnage_mt == 8
    assert result.accepted_records[0].contained_metal == 870_000
    assert result.candidate_records == []
    assert result.rounds[0].decision == "pass"


def test_progress_callback_reports_pipeline_stages() -> None:
    messages: list[str] = []

    result = NI43101Orchestrator(
        FakeExtractor([_extraction()]),
        FakeCritic([_critic_result(9)]),
    ).run_with_pages(
        "report-1",
        [_page()],
        progress_callback=messages.append,
    )

    assert result.status == "pass"
    assert any("DeepSeek" in message for message in messages)
    assert any("确定性校验" in message for message in messages)
    assert any("GLM CriticMaster" in message for message in messages)
    assert messages[-1].endswith("PASS")


def test_high_critic_score_cannot_override_deterministic_hard_fail() -> None:
    extractor = FakeExtractor([_extraction(_raw_record(metal_value=600))])
    critic = FakeCritic([_critic_result(9)])

    result = NI43101Orchestrator(
        extractor,
        critic,
        max_revise_rounds=1,
    ).run_with_pages("report-1", [_page()])

    assert result.status == "abstain"
    assert result.accepted_records == []
    assert len(result.candidate_records) == 1
    assert result.rounds[0].validator_results[0].status == "HARD_FAIL"
    assert "deterministic HARD_FAIL" in result.abstain_reason
    evolution_event = orjson.loads(
        Path("evolution.jsonl").read_bytes().splitlines()[0]
    )
    assert evolution_event["decision"] == "abstain"
    assert "MATH_INCONSISTENCY" in evolution_event["failure_types"]


def test_three_non_passing_rounds_end_in_abstain() -> None:
    extractor = FakeExtractor([_extraction(), _extraction(), _extraction()])
    critic = FakeCritic(
        [
            _critic_result(5, "revise"),
            _critic_result(6, "revise"),
            _critic_result(7, "revise"),
        ]
    )

    result = NI43101Orchestrator(extractor, critic).run_with_pages(
        "report-1",
        [_page()],
    )

    assert result.status == "abstain"
    assert result.needs_human_review is True
    assert [round_result.decision for round_result in result.rounds] == [
        "revise",
        "revise",
        "abstain",
    ]
    assert [round_result.critic_result.score for round_result in result.rounds] == [
        5,
        6,
        7,
    ]
    assert extractor.extract_calls == 1
    assert len(extractor.revision_calls) == 2
    assert critic.calls == 3
    evolution_lines = Path("evolution.jsonl").read_bytes().splitlines()
    assert len(evolution_lines) == 3
    assert [orjson.loads(line)["decision"] for line in evolution_lines] == [
        "revise",
        "revise",
        "abstain",
    ]


def test_uncertain_extraction_cannot_pass_with_score_nine() -> None:
    extractor = FakeExtractor([_extraction(uncertain=True)])
    critic = FakeCritic([_critic_result(9)])

    result = NI43101Orchestrator(
        extractor,
        critic,
        max_revise_rounds=1,
    ).run_with_pages("report-1", [_page()])

    assert result.status == "abstain"
    assert result.accepted_records == []
    assert len(result.candidate_records) == 1
    assert result.candidate_records[0].location == "Example deposit"
    assert "extractor uncertainty" in result.abstain_reason


def test_unresolved_critical_issue_blocks_high_score() -> None:
    critical = CriticIssue(
        code="evidence_not_supported",
        severity="critical",
        record_index=0,
        field="metal_value",
        message="The source row does not support the extracted number.",
        source_evidence="Table row is unreadable.",
        suggested_action="Re-check the original table or abstain.",
    )
    critic_result = _critic_result(9).model_copy(update={"issues": [critical]})
    result = NI43101Orchestrator(
        FakeExtractor([_extraction()]),
        FakeCritic([critic_result]),
        max_revise_rounds=1,
    ).run_with_pages("report-1", [_page()])

    assert result.status == "abstain"
    assert result.accepted_records == []
    assert "evidence_not_supported" in result.abstain_reason
    evolution_event = orjson.loads(
        Path("evolution.jsonl").read_bytes().splitlines()[0]
    )
    assert evolution_event["decision"] == "abstain"
    assert "MISSING_EVIDENCE" in evolution_event["failure_types"]


def test_contained_copper_mt_suffix_is_normalized_semantically() -> None:
    raw = _raw_record(
        commodity="Cu",
        tonnage_value=1700,
        tonnage_unit="Mt",
        grade_value=1,
        grade_unit="% Cu",
        metal_value=17,
        metal_unit="Mt Cu",
    )
    extractor = FakeExtractor([_extraction(raw)])
    critic = FakeCritic([_critic_result(9)])

    result = NI43101Orchestrator(extractor, critic).run_with_pages(
        "report-1",
        [_page()],
    )

    normalized = result.accepted_records[0]
    assert normalized.tonnage_mt == 1700
    assert normalized.contained_metal == 17_000_000
    assert normalized.metal_unit == "t"
    assert normalized.source_values["metal_unit"] == "Mt Cu"


def test_empty_extraction_cannot_pass_even_when_critic_returns_nine() -> None:
    empty = ExtractionResult(
        document_id="report-1",
        candidate_pages=[10],
        records=[],
        uncertain=False,
        uncertainty_reasons=[],
        input_mode="candidate_pages_with_context",
    )

    result = NI43101Orchestrator(
        FakeExtractor([empty]),
        FakeCritic([_critic_result(9)]),
        max_revise_rounds=1,
    ).run_with_pages("report-1", [_page()])

    assert result.status == "abstain"
    assert result.accepted_records == []
    assert result.needs_human_review is True
    assert "no target Indicated/Inferred records" in result.abstain_reason


def test_extractor_format_failure_is_logged_then_reraised() -> None:
    class BrokenExtractor:
        def extract(self, *args: object, **kwargs: object) -> ExtractionResult:
            raise ExtractorOutputError("report-1", "invalid JSON")

    with pytest.raises(ExtractorOutputError, match="invalid JSON"):
        NI43101Orchestrator(
            BrokenExtractor(),
            FakeCritic([]),
        ).run_with_pages("report-1", [_page()], case_id="format-case")

    event = orjson.loads(Path("evolution.jsonl").read_bytes().splitlines()[0])
    assert event["case_id"] == "format-case"
    assert event["decision"] == "llm_format_error"
    assert event["failure_types"] == ["LLM_FORMAT_ERROR"]
    assert event["extractor_output"]["stage"] == "extractor"


def test_critic_format_failure_is_logged_then_reraised() -> None:
    class BrokenCritic:
        def review(self, *args: object, **kwargs: object) -> CriticResult:
            raise CriticOutputError("invalid CriticResult JSON")

    with pytest.raises(CriticOutputError, match="invalid CriticResult JSON"):
        NI43101Orchestrator(
            FakeExtractor([_extraction()]),
            BrokenCritic(),
        ).run_with_pages("report-1", [_page()], case_id="critic-format-case")

    event = orjson.loads(Path("evolution.jsonl").read_bytes().splitlines()[0])
    assert event["decision"] == "llm_format_error"
    assert event["critic_output"]["stage"] == "critic"
    assert event["extractor_output"]["records"]
