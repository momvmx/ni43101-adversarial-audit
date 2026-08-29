"""Append-only, non-blocking evolution log for failed audit decisions."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ni43101.critic import CriticResult
from ni43101.schemas import Commodity, ExtractionResult, ResourceCategory
from ni43101.validators import DeterministicValidationResult


DEFAULT_EVOLUTION_PATH = Path("evolution.jsonl")


class FailureType(str, Enum):
    RESOURCE_RESERVE_CONFUSION = "RESOURCE_RESERVE_CONFUSION"
    CATEGORY_COLUMN_SHIFT = "CATEGORY_COLUMN_SHIFT"
    MEASURED_PLUS_INDICATED_CONFUSION = (
        "MEASURED_PLUS_INDICATED_CONFUSION"
    )
    ROW_ALIGNMENT_ERROR = "ROW_ALIGNMENT_ERROR"
    TONNAGE_UNIT_ERROR = "TONNAGE_UNIT_ERROR"
    METAL_UNIT_ERROR = "METAL_UNIT_ERROR"
    COMMODITY_MISMATCH = "COMMODITY_MISMATCH"
    MATH_INCONSISTENCY = "MATH_INCONSISTENCY"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    AMBIGUOUS_TABLE = "AMBIGUOUS_TABLE"
    LLM_FORMAT_ERROR = "LLM_FORMAT_ERROR"
    MISSING_CATEGORY = "MISSING_CATEGORY"
    LOW_CRITIC_SCORE = "LOW_CRITIC_SCORE"


LESSONS: dict[FailureType, str] = {
    FailureType.RESOURCE_RESERVE_CONFUSION: (
        "Reject tables dominated by Proven/Probable or Mineral Reserve Statement "
        "when the task requires Indicated/Inferred Mineral Resources."
    ),
    FailureType.CATEGORY_COLUMN_SHIFT: (
        "Align each value triplet with its exact category header before extracting "
        "Indicated or Inferred records."
    ),
    FailureType.MEASURED_PLUS_INDICATED_CONFUSION: (
        "When a table contains both Indicated and Measured + Indicated, align every "
        "tonnage-grade-metal triplet with its exact parent header before extraction."
    ),
    FailureType.ROW_ALIGNMENT_ERROR: (
        "Take location, tonnage, grade, and contained metal from the same source row "
        "and category group."
    ),
    FailureType.TONNAGE_UNIT_ERROR: (
        "Preserve the original tonnage unit before normalization. Convert kt to Mt "
        "only after extraction."
    ),
    FailureType.METAL_UNIT_ERROR: (
        "Do not treat contained metal units as ore tonnage units. For gold normalize "
        "koz/Moz to oz. For copper normalize kt/Mt Cu to tonnes."
    ),
    FailureType.COMMODITY_MISMATCH: (
        "Keep commodity sections separate: Au requires g/t Au and oz, while Cu "
        "requires % Cu and tonnes."
    ),
    FailureType.MATH_INCONSISTENCY: (
        "Re-check tonnage, grade, contained metal, units, and column alignment when "
        "the deterministic relative error exceeds tolerance."
    ),
    FailureType.MISSING_EVIDENCE: (
        "Accept a numeric field only when the supplied source page and evidence text "
        "directly support it."
    ),
    FailureType.AMBIGUOUS_TABLE: (
        "When headers or rows remain ambiguous, preserve the uncertainty and abstain "
        "instead of guessing."
    ),
    FailureType.LLM_FORMAT_ERROR: (
        "Require schema-valid JSON and treat malformed model output as a controlled "
        "failure rather than partial extraction."
    ),
    FailureType.MISSING_CATEGORY: (
        "Check the complete target table for both Indicated and Inferred groups "
        "before accepting the extraction."
    ),
    FailureType.LOW_CRITIC_SCORE: (
        "Do not accept an extraction below the configured Critic score threshold; "
        "revise from original evidence or abstain."
    ),
}


EvolutionDecision = Literal[
    "revise",
    "abstain",
    "validator_hard_fail",
    "critic_critical",
    "evaluation_failed",
    "llm_format_error",
]


class EvolutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    case_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    pdf_name: str = Field(min_length=1)
    extractor_model: str = Field(min_length=1)
    critic_model: str = Field(min_length=1)
    round_number: int = Field(ge=0)
    candidate_pages: list[int]
    commodity: Commodity | list[Commodity] | None
    category: ResourceCategory | list[ResourceCategory] | None
    failure_types: list[FailureType] = Field(min_length=1)
    extractor_output: dict[str, Any] | None
    validator_output: list[dict[str, Any]]
    critic_output: dict[str, Any] | None
    decision: EvolutionDecision
    lesson: str = Field(min_length=1)
    ground_truth_diff: Any | None = None


def lesson_for_failure_types(failure_types: Iterable[FailureType]) -> str:
    """Return stable, deduplicated lessons in failure order."""

    lessons: list[str] = []
    for failure_type in failure_types:
        lesson = LESSONS[failure_type]
        if lesson not in lessons:
            lessons.append(lesson)
    return " ".join(lessons)


def infer_failure_types(
    extraction: ExtractionResult,
    validator_results: Sequence[DeterministicValidationResult],
    critic_result: CriticResult,
    *,
    pass_score: float,
) -> list[FailureType]:
    """Classify one audited round without mutating any supplied result."""

    failures: list[FailureType] = []

    def add(failure_type: FailureType) -> None:
        if failure_type not in failures:
            failures.append(failure_type)

    if critic_result.score < pass_score:
        add(FailureType.LOW_CRITIC_SCORE)
    if extraction.uncertain or any(record.uncertain for record in extraction.records):
        add(FailureType.AMBIGUOUS_TABLE)

    for validation in validator_results:
        if validation.status != "HARD_FAIL":
            continue
        text = " ".join(validation.messages).casefold()
        if validation.relative_error is not None and validation.relative_error > 0.10:
            add(FailureType.MATH_INCONSISTENCY)
        if "ore tonnage" in text or "tonnage unit" in text:
            add(FailureType.TONNAGE_UNIT_ERROR)
        if any(token in text for token in ("gold metal", "copper metal", "metal unit")):
            add(FailureType.METAL_UNIT_ERROR)
        if "incompatible" in text or "commodity" in text or "grade unit" in text:
            add(FailureType.COMMODITY_MISMATCH)

    issue_text = " ".join(
        f"{issue.code} {issue.field or ''} {issue.message}"
        for issue in critic_result.issues
        if issue.severity == "critical"
    ).casefold()
    if "measured_plus_indicated" in issue_text or "measured + indicated" in issue_text:
        add(FailureType.MEASURED_PLUS_INDICATED_CONFUSION)
    elif critic_result.checks.category_alignment == "critical":
        add(FailureType.CATEGORY_COLUMN_SHIFT)

    check_mapping = {
        "resource_vs_reserve": FailureType.RESOURCE_RESERVE_CONFUSION,
        "row_alignment": FailureType.ROW_ALIGNMENT_ERROR,
        "commodity_alignment": FailureType.COMMODITY_MISMATCH,
        "math_consistency": FailureType.MATH_INCONSISTENCY,
        "evidence_support": FailureType.MISSING_EVIDENCE,
        "completeness": FailureType.MISSING_CATEGORY,
    }
    checks = critic_result.checks.model_dump()
    for check_name, failure_type in check_mapping.items():
        if checks[check_name] == "critical":
            add(failure_type)

    if "reserve" in issue_text or "proven" in issue_text or "probable" in issue_text:
        add(FailureType.RESOURCE_RESERVE_CONFUSION)
    if "row" in issue_text or "alignment" in issue_text and "category" not in issue_text:
        add(FailureType.ROW_ALIGNMENT_ERROR)
    if "tonnage" in issue_text and "unit" in issue_text:
        add(FailureType.TONNAGE_UNIT_ERROR)
    if "metal" in issue_text and "unit" in issue_text:
        add(FailureType.METAL_UNIT_ERROR)
    if "commodity" in issue_text:
        add(FailureType.COMMODITY_MISMATCH)
    if "math" in issue_text or "relative_error" in issue_text:
        add(FailureType.MATH_INCONSISTENCY)
    if "evidence" in issue_text or "unsupported" in issue_text:
        add(FailureType.MISSING_EVIDENCE)
    if "missing" in issue_text and any(
        category in issue_text for category in ("indicated", "inferred", "category")
    ):
        add(FailureType.MISSING_CATEGORY)

    if not failures:
        add(FailureType.AMBIGUOUS_TABLE)
    return failures


def build_round_record(
    *,
    case_id: str,
    document_id: str,
    pdf_name: str,
    extractor_model: str,
    critic_model: str,
    round_number: int,
    extraction: ExtractionResult,
    validator_results: Sequence[DeterministicValidationResult],
    critic_result: CriticResult,
    decision: Literal["revise", "abstain"],
    pass_score: float,
) -> EvolutionRecord:
    failure_types = infer_failure_types(
        extraction,
        validator_results,
        critic_result,
        pass_score=pass_score,
    )
    commodities = sorted({record.commodity for record in extraction.records})
    categories = sorted({record.category for record in extraction.records})
    return EvolutionRecord(
        case_id=case_id,
        document_id=document_id,
        pdf_name=pdf_name,
        extractor_model=extractor_model,
        critic_model=critic_model,
        round_number=round_number,
        candidate_pages=list(extraction.candidate_pages),
        commodity=_single_or_list(commodities),
        category=_single_or_list(categories),
        failure_types=failure_types,
        extractor_output=extraction.model_dump(mode="json"),
        validator_output=[
            result.model_dump(mode="json") for result in validator_results
        ],
        critic_output=critic_result.model_dump(mode="json"),
        decision=decision,
        lesson=lesson_for_failure_types(failure_types),
    )


def build_evaluation_failure_record(
    *,
    case_id: str,
    document_id: str,
    pdf_name: str,
    extractor_model: str,
    critic_model: str,
    round_number: int,
    candidate_pages: Sequence[int],
    failure_types: Sequence[FailureType],
    ground_truth_diff: Any,
    extractor_output: dict[str, Any] | None = None,
    validator_output: Sequence[dict[str, Any]] = (),
    critic_output: dict[str, Any] | None = None,
) -> EvolutionRecord:
    """Build an evaluator-owned event; ground truth never enters Critic input."""

    resolved_failures = list(dict.fromkeys(failure_types))
    if not resolved_failures:
        raise ValueError("evaluation failure requires at least one failure type")
    return EvolutionRecord(
        case_id=case_id,
        document_id=document_id,
        pdf_name=pdf_name,
        extractor_model=extractor_model,
        critic_model=critic_model,
        round_number=round_number,
        candidate_pages=list(candidate_pages),
        commodity=None,
        category=None,
        failure_types=resolved_failures,
        extractor_output=extractor_output,
        validator_output=list(validator_output),
        critic_output=critic_output,
        decision="evaluation_failed",
        lesson=lesson_for_failure_types(resolved_failures),
        ground_truth_diff=ground_truth_diff,
    )


def build_llm_format_error_record(
    *,
    case_id: str,
    document_id: str,
    pdf_name: str,
    extractor_model: str,
    critic_model: str,
    round_number: int,
    candidate_pages: Sequence[int],
    stage: Literal["extractor", "critic"],
    extraction: ExtractionResult | None = None,
    validator_results: Sequence[DeterministicValidationResult] = (),
) -> EvolutionRecord:
    """Build a secret-free event for invalid model structured output."""

    failure_types = [FailureType.LLM_FORMAT_ERROR]
    commodities = sorted(
        {record.commodity for record in extraction.records}
        if extraction is not None
        else set()
    )
    categories = sorted(
        {record.category for record in extraction.records}
        if extraction is not None
        else set()
    )
    extractor_output = (
        extraction.model_dump(mode="json")
        if extraction is not None
        else {"stage": "extractor", "status": "invalid_structured_output"}
    )
    critic_output = (
        {"stage": "critic", "status": "invalid_structured_output"}
        if stage == "critic"
        else None
    )
    return EvolutionRecord(
        case_id=case_id,
        document_id=document_id,
        pdf_name=pdf_name,
        extractor_model=extractor_model,
        critic_model=critic_model,
        round_number=round_number,
        candidate_pages=list(candidate_pages),
        commodity=_single_or_list(commodities),
        category=_single_or_list(categories),
        failure_types=failure_types,
        extractor_output=extractor_output,
        validator_output=[
            result.model_dump(mode="json") for result in validator_results
        ],
        critic_output=critic_output,
        decision="llm_format_error",
        lesson=lesson_for_failure_types(failure_types),
    )


class EvolutionLogger:
    """Best-effort append-only JSONL writer and tolerant reader."""

    def __init__(self, path: str | Path = DEFAULT_EVOLUTION_PATH) -> None:
        self.path = Path(path)

    def append(self, record: EvolutionRecord) -> bool:
        """Append one complete JSON line; return False instead of raising on I/O."""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = orjson.dumps(record.model_dump(mode="json")) + b"\n"
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
            return True
        except (OSError, TypeError, ValueError) as error:
            warnings.warn(
                f"evolution log append failed for {self.path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def log_evaluation_failure(
        self,
        *,
        case_id: str,
        document_id: str,
        pdf_name: str,
        extractor_model: str,
        critic_model: str,
        round_number: int,
        candidate_pages: Sequence[int],
        failure_types: Sequence[FailureType],
        ground_truth_diff: Any,
        extractor_output: dict[str, Any] | None = None,
        validator_output: Sequence[dict[str, Any]] = (),
        critic_output: dict[str, Any] | None = None,
    ) -> bool:
        """Build and append one evaluator-owned failure event."""

        try:
            record = build_evaluation_failure_record(
                case_id=case_id,
                document_id=document_id,
                pdf_name=pdf_name,
                extractor_model=extractor_model,
                critic_model=critic_model,
                round_number=round_number,
                candidate_pages=candidate_pages,
                failure_types=failure_types,
                ground_truth_diff=ground_truth_diff,
                extractor_output=extractor_output,
                validator_output=validator_output,
                critic_output=critic_output,
            )
        except (TypeError, ValueError, ValidationError) as error:
            warnings.warn(
                f"evolution evaluation record failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        return self.append(record)

    def log_llm_format_error(
        self,
        *,
        case_id: str,
        document_id: str,
        pdf_name: str,
        extractor_model: str,
        critic_model: str,
        round_number: int,
        candidate_pages: Sequence[int],
        stage: Literal["extractor", "critic"],
        extraction: ExtractionResult | None = None,
        validator_results: Sequence[DeterministicValidationResult] = (),
    ) -> bool:
        """Append a format-failure event without storing provider error text."""

        try:
            record = build_llm_format_error_record(
                case_id=case_id,
                document_id=document_id,
                pdf_name=pdf_name,
                extractor_model=extractor_model,
                critic_model=critic_model,
                round_number=round_number,
                candidate_pages=candidate_pages,
                stage=stage,
                extraction=extraction,
                validator_results=validator_results,
            )
        except (TypeError, ValueError, ValidationError) as error:
            warnings.warn(
                f"evolution LLM format record failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        return self.append(record)

    def read(self) -> list[EvolutionRecord]:
        """Read valid records and skip malformed JSON/records with a warning."""

        records: list[EvolutionRecord] = []
        try:
            lines = self.path.read_bytes().splitlines()
        except FileNotFoundError:
            return records
        except OSError as error:
            warnings.warn(
                f"evolution log read failed for {self.path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return records

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvolutionRecord.model_validate(orjson.loads(line)))
            except (orjson.JSONDecodeError, ValidationError, TypeError) as error:
                warnings.warn(
                    f"skipping invalid evolution log line {line_number}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return records


def append_evolution_record(
    record: EvolutionRecord,
    path: str | Path = DEFAULT_EVOLUTION_PATH,
) -> bool:
    return EvolutionLogger(path).append(record)


def read_evolution_log(
    path: str | Path = DEFAULT_EVOLUTION_PATH,
) -> list[EvolutionRecord]:
    return EvolutionLogger(path).read()


def _single_or_list(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values
