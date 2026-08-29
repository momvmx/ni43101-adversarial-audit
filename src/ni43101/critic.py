"""GLM-4.7-Flash adversarial CriticMaster for extraction audits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field

from ni43101.config import Settings, get_settings
from ni43101.llm_client import (
    FakeLLMClient,
    LLMClientError,
    OpenAICompatibleClient,
    StructuredOutputError,
    StructuredCompletionClient,
)
from ni43101.pdf_locator import CandidatePage
from ni43101.schemas import ExtractionResult, ResourceRecord
from ni43101.validators import (
    DeterministicValidationResult,
    MetalConsistencyResult,
)


IssueSeverity = Literal["info", "warning", "critical"]
CriticVerdict = Literal["pass", "revise", "abstain"]
CheckStatus = Literal["pass", "warning", "critical"]
ValidationEvidence = MetalConsistencyResult | DeterministicValidationResult


class _CriticSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CriticIssue(_CriticSchema):
    code: str = Field(min_length=1)
    severity: IssueSeverity
    record_index: int | None = Field(default=None, ge=0)
    field: str | None = None
    message: str = Field(min_length=1)
    source_evidence: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)


class CriticChecks(_CriticSchema):
    resource_vs_reserve: CheckStatus
    category_alignment: CheckStatus
    row_alignment: CheckStatus
    commodity_alignment: CheckStatus
    unit_correctness: CheckStatus
    math_consistency: CheckStatus
    evidence_support: CheckStatus
    completeness: CheckStatus


class CriticResult(_CriticSchema):
    score: int = Field(ge=1, le=10)
    verdict: CriticVerdict
    issues: list[CriticIssue]
    checks: CriticChecks
    summary: str = Field(min_length=1)


CRITIC_SYSTEM_PROMPT = """You are the adversarial CriticMaster for NI 43-101 Mineral Resource extraction. You are not a second extractor. Your job is to
actively find unsafe assumptions, column shifts, unsupported values, omissions,
and semantic errors in a DeepSeek extraction.

When evidence is ambiguous, prefer revise or abstain over unsafe acceptance. Do
not try to help the extractor reach a passing score. Do not invent or repair
numbers. Judge only the supplied source pages, extraction, normalized records,
and deterministic validator results. No evaluator reference answers are
provided or permitted.

Audit all eight dimensions:

1. resource_vs_reserve
Distinguish Mineral Resources from Mineral Reserves. Proven or Probable usually
indicates a Reserve table. Treat extraction of a Reserve as a Resource as
critical.

2. category_alignment
Verify the Measured, Indicated, Measured + Indicated, and Inferred column groups.
Confusing Measured + Indicated with Indicated is critical.

3. row_alignment
Verify location, tonnage, grade, and contained metal come from the same row and
the same category.

4. commodity_alignment
Gold must use g/t Au and oz-based contained metal. Copper must use % Cu and
tonne-based contained metal. Never combine Cu tonnage, Au grade, and Cu metal,
or accept Cu + oz, Au + % Cu, Au + t, or Cu + g/t Au.

5. unit_correctness
Check t/kt/Mt ore conversions, oz/koz/Moz gold conversions, and t/kt/Mt Cu
conversions. Distinguish ore Mt from contained Cu Mt.

6. math_consistency
Use the supplied Python validator relative_error. At or below 5% passes; above
5% through 10% is a warning; above 10% is critical. Never override a Python
HARD_FAIL with your own opinion.

7. evidence_support
Every number must be supported by the supplied source page. Plausibility is not
evidence. Unsupported or fabricated core values are critical.

8. completeness
Flag a missing Indicated or Inferred target when the table clearly contains it.
If the evidence is ambiguous, abstain rather than supplying missing numbers.

Scoring: 10 means exceptionally clear evidence and all deterministic checks
pass; 8-9 means reliable with only rounding or formatting issues; 6-7 means a
real risk requiring revision; 3-5 means likely wrong; 1-2 means severe error.
Return only CriticResult JSON and no Markdown."""


CRITIC_TASK = """Adversarially audit the DeepSeek raw extraction against the
candidate source text, normalized records, and deterministic validator results.
Report concrete issues only, cite source evidence, complete all eight checks,
and choose pass, revise, or abstain without inventing replacement values."""


class CriticError(RuntimeError):
    """Controlled CriticMaster failure suitable for handling by a pipeline."""


class CriticConfigurationError(CriticError):
    """Raised when CriticMaster configuration is incomplete or unsafe."""


class CriticOutputError(CriticError):
    """Raised when CriticMaster returns malformed or out-of-scope output."""


class FakeCriticClient(FakeLLMClient):
    """Explicit fake client for CriticMaster unit tests."""


class CriticMaster:
    """Run a GLM adversarial review and enforce deterministic hard gates."""

    def __init__(self, client: StructuredCompletionClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CriticMaster:
        resolved = settings or get_settings()
        if resolved.critic_api_key is None:
            raise CriticConfigurationError("CRITIC_API_KEY is not configured")
        if not resolved.critic_base_url:
            raise CriticConfigurationError("CRITIC_BASE_URL is not configured")
        if resolved.critic_model.casefold() == resolved.extractor_model.casefold():
            raise CriticConfigurationError(
                "Extractor and CriticMaster must use different models"
            )

        return cls(
            OpenAICompatibleClient(
                api_key=resolved.critic_api_key.get_secret_value(),
                base_url=resolved.critic_base_url,
                model=resolved.critic_model,
            )
        )

    def review(
        self,
        candidate_pages: Sequence[CandidatePage],
        raw_extraction: ExtractionResult,
        normalized_records: Sequence[ResourceRecord],
        validator_results: Sequence[ValidationEvidence],
    ) -> CriticResult:
        self._validate_inputs(
            raw_extraction,
            normalized_records,
            validator_results,
        )
        messages = self._build_messages(
            candidate_pages,
            raw_extraction,
            normalized_records,
            validator_results,
        )

        try:
            llm_result = self.client.structured_completion(messages, CriticResult)
        except StructuredOutputError as error:
            raise CriticOutputError(
                f"CriticMaster returned invalid structured output: {error}"
            ) from error
        except LLMClientError as error:
            raise CriticError(f"CriticMaster request failed: {error}") from error

        self._validate_issue_indexes(llm_result, len(raw_extraction.records))
        return self._apply_deterministic_gates(llm_result, validator_results)

    @staticmethod
    def _validate_inputs(
        raw_extraction: ExtractionResult,
        normalized_records: Sequence[ResourceRecord],
        validator_results: Sequence[ValidationEvidence],
    ) -> None:
        for index, record in enumerate(normalized_records):
            if record.document_id != raw_extraction.document_id:
                raise CriticError(
                    f"normalized record {index} document_id does not match extraction"
                )

        if all(
            isinstance(result, MetalConsistencyResult)
            for result in validator_results
        ):
            if len(normalized_records) != len(validator_results):
                raise CriticError(
                    "normalized_records and validator_results must have equal length"
                )
            return

        seen_raw_indexes: set[int] = set()
        for result in validator_results:
            if not isinstance(result, DeterministicValidationResult):
                raise CriticError("validator result formats cannot be mixed")
            if result.record_index >= len(raw_extraction.records):
                raise CriticError(
                    f"validator result cites unknown raw record {result.record_index}"
                )
            if result.record_index in seen_raw_indexes:
                raise CriticError(
                    f"duplicate validator result for raw record {result.record_index}"
                )
            seen_raw_indexes.add(result.record_index)
            if (
                result.normalized_record_index is not None
                and result.normalized_record_index >= len(normalized_records)
            ):
                raise CriticError(
                    "validator result cites unknown normalized record "
                    f"{result.normalized_record_index}"
                )

        if seen_raw_indexes != set(range(len(raw_extraction.records))):
            raise CriticError("every raw record requires one validator result")

    @classmethod
    def _build_messages(
        cls,
        candidate_pages: Sequence[CandidatePage],
        raw_extraction: ExtractionResult,
        normalized_records: Sequence[ResourceRecord],
        validator_results: Sequence[ValidationEvidence],
    ) -> list[dict[str, str]]:
        pages_by_number = {
            page.page_number: {
                "page_number": page.page_number,
                "text": page.text,
            }
            for page in candidate_pages
        }
        payload = {
            "task": CRITIC_TASK,
            "candidate_pages": [
                pages_by_number[number] for number in sorted(pages_by_number)
            ],
            "deepseek_raw_extraction": raw_extraction.model_dump(mode="json"),
            "normalized_records": [
                record.model_dump(mode="json") for record in normalized_records
            ],
            "deterministic_validator_results": [
                cls._validator_payload(index, result)
                for index, result in enumerate(validator_results)
            ],
            "output_schema": CriticResult.model_json_schema(),
        }
        return [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode("utf-8"),
            },
        ]

    @staticmethod
    def _validate_issue_indexes(result: CriticResult, record_count: int) -> None:
        for issue in result.issues:
            if issue.record_index is not None and issue.record_index >= record_count:
                raise CriticOutputError(
                    f"CriticMaster cited unknown record index {issue.record_index}"
                )

    @staticmethod
    def _validator_payload(
        fallback_index: int,
        result: ValidationEvidence,
    ) -> dict[str, object]:
        payload = result.model_dump(mode="json")
        payload.setdefault("record_index", fallback_index)
        return payload

    @classmethod
    def _apply_deterministic_gates(
        cls,
        result: CriticResult,
        validator_results: Sequence[ValidationEvidence],
    ) -> CriticResult:
        score = result.score
        verdict = result.verdict
        issues = list(result.issues)
        checks = result.checks.model_copy(deep=True)

        hard_fail_results = [
            (cls._record_index(index, validation), validation)
            for index, validation in enumerate(validator_results)
            if validation.status == "HARD_FAIL"
            or cls._relative_error(validation) > 0.10
        ]
        hard_fail_index_set = {index for index, _ in hard_fail_results}
        warning_results = [
            (cls._record_index(index, validation), validation)
            for index, validation in enumerate(validator_results)
            if cls._record_index(index, validation) not in hard_fail_index_set
            and (
                validation.status == "WARNING"
                or 0.05 < cls._relative_error(validation) <= 0.10
            )
        ]

        if warning_results:
            score = min(score, 9)
            checks.math_consistency = cls._max_check_status(
                checks.math_consistency,
                "warning",
            )
            for index, validation in warning_results:
                if not cls._has_issue(issues, "math_consistency_warning", index):
                    issues.append(
                        CriticIssue(
                            code="math_consistency_warning",
                            severity="warning",
                            record_index=index,
                            field="contained_metal",
                            message=(
                                "Reported contained metal differs from the "
                                "grade-derived value by more than 5%"
                            ),
                            source_evidence=cls._validation_evidence(validation),
                            suggested_action=(
                                "Verify table rounding, units, and column alignment"
                            ),
                        )
                    )

        if hard_fail_results:
            score = min(score, 7)
            if verdict == "pass":
                verdict = "revise"
            for index, validation in hard_fail_results:
                math_failure = cls._relative_error(validation) > 0.10
                code = (
                    "math_consistency_hard_fail"
                    if math_failure
                    else "deterministic_hard_fail"
                )
                field = "contained_metal" if math_failure else None
                if math_failure:
                    checks.math_consistency = "critical"
                    message = (
                        "Contained metal relative error exceeds 10%; deterministic "
                        "validation is a HARD_FAIL"
                    )
                    action = (
                        "Re-check tonnage, grade, contained metal, units, and column "
                        "alignment before acceptance"
                    )
                else:
                    checks.commodity_alignment = "critical"
                    checks.unit_correctness = "critical"
                    message = (
                        "Deterministic validation is a HARD_FAIL despite a math "
                        "error not exceeding 10%, indicating a unit or commodity "
                        "compatibility failure"
                    )
                    action = "Correct the commodity and unit combination, then revalidate"

                if not cls._has_issue(issues, code, index):
                    issues.append(
                        CriticIssue(
                            code=code,
                            severity="critical",
                            record_index=index,
                            field=field,
                            message=message,
                            source_evidence=cls._validation_evidence(validation),
                            suggested_action=action,
                        )
                    )

        if any(issue.severity == "critical" for issue in issues):
            score = min(score, 7)
            if verdict == "pass":
                verdict = "revise"

        if score <= 7 and verdict == "pass":
            verdict = "revise"

        return result.model_copy(
            update={
                "score": score,
                "verdict": verdict,
                "issues": issues,
                "checks": checks,
            }
        )

    @staticmethod
    def _validation_evidence(validation: ValidationEvidence) -> str:
        expected = "unknown" if validation.expected is None else f"{validation.expected:g}"
        actual = "unknown" if validation.actual is None else f"{validation.actual:g}"
        error = (
            "unknown"
            if validation.relative_error is None
            else f"{validation.relative_error:.6f}"
        )
        return (
            f"expected={expected}, actual={actual}, relative_error={error}, "
            f"status={validation.status}"
        )

    @staticmethod
    def _record_index(fallback_index: int, validation: ValidationEvidence) -> int:
        if isinstance(validation, DeterministicValidationResult):
            return validation.record_index
        return fallback_index

    @staticmethod
    def _relative_error(validation: ValidationEvidence) -> float:
        return validation.relative_error or 0.0

    @staticmethod
    def _has_issue(
        issues: Sequence[CriticIssue],
        code: str,
        record_index: int,
    ) -> bool:
        return any(
            issue.code == code and issue.record_index == record_index
            for issue in issues
        )

    @staticmethod
    def _max_check_status(current: CheckStatus, minimum: CheckStatus) -> CheckStatus:
        rank = {"pass": 0, "warning": 1, "critical": 2}
        return current if rank[current] >= rank[minimum] else minimum
