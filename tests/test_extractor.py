import orjson
import pytest

import ni43101.extractor as extractor_module
from ni43101.extractor import (
    EXTRACTOR_REVISION_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    DeepSeekExtractor,
    ExtractorError,
)
from ni43101.llm_client import FakeLLMClient
from ni43101.pdf_locator import CandidatePage


def _page(index: int, score: float, text: str | None = None) -> CandidatePage:
    return CandidatePage(
        page_index=index,
        page_number=index + 1,
        score=score,
        matched_positive_keywords=[],
        matched_negative_keywords=[],
        reasons=[],
        text=text or f"page {index + 1}",
    )


def _raw_record(
    *,
    category: str,
    uncertain: bool = False,
    uncertainty_reason: str | None = None,
) -> dict[str, object]:
    return {
        "document_id": "report-1",
        "location": "Tanami UG",
        "category": category,
        "commodity": "Au",
        "tonnage_value": 8000,
        "tonnage_unit": "kt",
        "grade_value": 3.4,
        "grade_unit": "g/t Au",
        "metal_value": 870,
        "metal_unit": "koz",
        "source_page": 10,
        "table_title": "Mineral Resource Statement",
        "evidence_text": "Indicated 8,000 kt 3.40 g/t Au 870 koz",
        "confidence": 0.95,
        "uncertain": uncertain,
        "uncertainty_reason": uncertainty_reason,
    }


def _result(records: list[dict[str, object]], **changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "document_id": "report-1",
        "candidate_pages": [10],
        "records": records,
        "uncertain": False,
        "uncertainty_reasons": [],
        "input_mode": "candidate_pages_with_context",
    }
    result.update(changes)
    return result


def test_extracts_normal_indicated_and_inferred_records() -> None:
    indicated = _raw_record(category="Indicated")
    inferred = {
        **_raw_record(category="Inferred"),
        "tonnage_value": 6300,
        "grade_value": 5.67,
        "metal_value": 1140,
        "evidence_text": "Inferred 6,300 kt 5.67 g/t Au 1,140 koz",
    }
    fake = FakeLLMClient([_result([indicated, inferred])])
    extractor = DeepSeekExtractor(fake)

    result = extractor.extract("report-1", [_page(9, 100)])

    assert [record.category for record in result.records] == [
        "Indicated",
        "Inferred",
    ]
    assert result.records[0].tonnage_value == 8000
    assert result.records[0].metal_unit == "koz"


def test_measured_plus_indicated_fails_schema_validation() -> None:
    invalid = orjson.dumps(
        _result([_raw_record(category="Measured + Indicated")])
    ).decode("utf-8")
    extractor = DeepSeekExtractor(FakeLLMClient([invalid]))

    with pytest.raises(ExtractorError, match="ExtractionResult"):
        extractor.extract("report-1", [_page(9, 100)])


def test_invalid_json_becomes_controlled_extractor_error() -> None:
    extractor = DeepSeekExtractor(FakeLLMClient(["not json"]))

    with pytest.raises(ExtractorError) as captured:
        extractor.extract("report-1", [_page(9, 100)])

    assert captured.value.document_id == "report-1"


def test_uncertain_record_is_preserved_with_reason() -> None:
    uncertain_record = _raw_record(
        category="Indicated",
        uncertain=True,
        uncertainty_reason="column alignment is ambiguous",
    )
    fake = FakeLLMClient(
        [
            _result(
                [uncertain_record],
                uncertain=True,
                uncertainty_reasons=["column alignment is ambiguous"],
            )
        ]
    )

    result = DeepSeekExtractor(fake).extract("report-1", [_page(9, 100)])

    assert result.uncertain is True
    assert result.records[0].uncertain is True
    assert result.records[0].uncertainty_reason == "column alignment is ambiguous"


def test_prompt_uses_only_top_three_candidates_and_neighbor_context() -> None:
    candidates = [
        _page(9, 100, "candidate 10"),
        _page(19, 90, "candidate 20"),
        _page(29, 80, "candidate 30"),
        _page(39, 70, "candidate 40 must not be sent"),
    ]
    context = [
        _page(8, 0, "neighbor 9"),
        _page(9, 0, "duplicate candidate 10"),
        _page(10, 0, "neighbor 11"),
        _page(38, 0, "neighbor of excluded candidate must not be sent"),
    ]
    fake = FakeLLMClient([_result([])])

    DeepSeekExtractor(fake).extract("report-1", candidates, context)

    payload = orjson.loads(fake.calls[0]["messages"][1]["content"])
    assert [page["page_number"] for page in payload["candidate_pages"]] == [
        10,
        20,
        30,
    ]
    assert [page["page_number"] for page in payload["context_pages"]] == [9, 11]
    serialized = fake.calls[0]["messages"][1]["content"]
    assert "candidate 40 must not be sent" not in serialized
    assert "neighbor of excluded candidate must not be sent" not in serialized


def test_evolution_fewshot_is_injected_without_changing_output_schema() -> None:
    class FakeFewShotProvider:
        def render(self, document_id, candidate_pages):
            assert document_id == "report-1"
            assert len(candidate_pages) == 1
            return "Previous extraction mistake: category column shift."

    fake = FakeLLMClient([_result([])])
    extractor = DeepSeekExtractor(
        fake,
        fewshot_provider=FakeFewShotProvider(),
    )

    extractor.extract("report-1", [_page(9, 100)])

    payload = orjson.loads(fake.calls[0]["messages"][1]["content"])
    assert payload["evolution_fewshot"].startswith("Previous extraction mistake")
    assert payload["output_schema"]["title"] == "ExtractionResult"


def test_system_prompt_contains_required_scope_guards() -> None:
    required_phrases = [
        "You are a strict NI 43-101 Mineral Resource table extraction agent.",
        "Correctness is more important than completeness.",
        "Indicated Mineral Resources",
        "Inferred Mineral Resources",
        "Measured + Indicated",
        "Proven",
        "Probable",
        "Mineral Reserves",
        "Mineral Resources and Mineral Reserves are different concepts.",
    ]

    assert all(phrase in EXTRACTOR_SYSTEM_PROMPT for phrase in required_phrases)


def test_revision_rechecks_source_and_includes_audit_feedback() -> None:
    initial = _result([_raw_record(category="Indicated")])
    fake = FakeLLMClient([initial])
    previous = FakeLLMClient([initial]).structured_completion(
        [],
        extractor_module.ExtractionResult,
    )

    DeepSeekExtractor(fake).revise(
        "report-1",
        [_page(9, 100, "original source table")],
        [_page(8, 0, "original title and notes")],
        previous,
        [{"record_index": 0, "status": "HARD_FAIL"}],
        [{"code": "column_shift", "severity": "critical"}],
    )

    messages = fake.calls[0]["messages"]
    payload = orjson.loads(messages[1]["content"])
    assert EXTRACTOR_REVISION_PROMPT in messages[0]["content"]
    assert payload["previous_extraction"]["records"][0]["metal_unit"] == "koz"
    assert payload["validator_failures"][0]["status"] == "HARD_FAIL"
    assert payload["critic_issues"][0]["code"] == "column_shift"
    assert "original source table" in messages[1]["content"]
    assert "Do not blindly accept" in messages[0]["content"]


def test_extract_from_pdf_uses_locator_top_three_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_page(9, 100)]
    context = [_page(8, 0), _page(9, 100), _page(10, 0)]
    locator_calls: list[tuple[object, int]] = []

    def fake_locate(path: object, top_k: int) -> list[CandidatePage]:
        locator_calls.append((path, top_k))
        return candidates

    monkeypatch.setattr(extractor_module, "locate_candidate_pages", fake_locate)
    monkeypatch.setattr(
        extractor_module,
        "get_candidate_context",
        lambda path, pages, surrounding_pages: context,
    )
    fake = FakeLLMClient([_result([])])

    result = DeepSeekExtractor(fake).extract_from_pdf(
        "data/pdfs/example.pdf",
        document_id="report-1",
    )

    assert result.document_id == "report-1"
    assert locator_calls[0][1] == 3
