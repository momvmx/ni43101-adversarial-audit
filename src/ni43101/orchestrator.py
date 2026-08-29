"""Bounded NI 43-101 extraction, validation, and adversarial-audit pipeline."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ni43101.config import Settings, get_settings
from ni43101.critic import (
    CriticIssue,
    CriticMaster,
    CriticOutputError,
    CriticResult,
)
from ni43101.extractor import DeepSeekExtractor, ExtractorOutputError
from ni43101.evolution import EvolutionLogger, build_round_record
from ni43101.fewshot import FewShotProvider
from ni43101.pdf_locator import (
    CandidatePage,
    get_candidate_context,
    locate_candidate_pages,
)
from ni43101.schemas import ExtractionResult, RawResourceRecord, ResourceRecord
from ni43101.units import normalize_contained_metal, normalize_ore_tonnage
from ni43101.validators import (
    DeterministicValidationResult,
    validate_category,
    validate_commodity,
    validate_metal_consistency,
)


PipelineStatus = Literal["pass", "abstain"]
RoundDecision = Literal["pass", "revise", "abstain"]
ProgressCallback = Callable[[str], None]


class _PipelineSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoundResult(_PipelineSchema):
    round_number: int = Field(ge=1)
    extractor_result: ExtractionResult
    normalized_records: list[ResourceRecord]
    validator_results: list[DeterministicValidationResult]
    critic_result: CriticResult
    decision: RoundDecision


class PipelineResult(_PipelineSchema):
    document_id: str = Field(min_length=1)
    status: PipelineStatus
    needs_human_review: bool
    accepted_records: list[ResourceRecord]
    candidate_records: list[RawResourceRecord]
    rounds: list[RoundResult]
    final_score: int = Field(ge=1, le=10)
    abstain_reason: str | None = None


class ExtractorAgent(Protocol):
    def extract(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage] | None = None,
        *,
        input_mode: str = "candidate_pages_with_context",
    ) -> ExtractionResult: ...

    def revise(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage] | None,
        previous_extraction: ExtractionResult,
        validator_failures: Sequence[DeterministicValidationResult],
        critic_issues: Sequence[CriticIssue],
        *,
        input_mode: str = "candidate_pages_with_context_revision",
    ) -> ExtractionResult: ...


class CriticAgent(Protocol):
    def review(
        self,
        candidate_pages: Sequence[CandidatePage],
        raw_extraction: ExtractionResult,
        normalized_records: Sequence[ResourceRecord],
        validator_results: Sequence[DeterministicValidationResult],
    ) -> CriticResult: ...


class OrchestratorConfigurationError(ValueError):
    """Raised for an invalid bounded-pipeline configuration."""


class NI43101Orchestrator:
    """Execute at most ``max_revise_rounds`` extractor/critic rounds."""

    def __init__(
        self,
        extractor: ExtractorAgent,
        critic: CriticAgent,
        *,
        max_revise_rounds: int = 3,
        pass_score: float = 8,
        evolution_logger: EvolutionLogger | None = None,
        extractor_model: str = "deepseek-chat",
        critic_model: str = "GLM-4.7-Flash",
    ) -> None:
        if not 1 <= max_revise_rounds <= 3:
            raise OrchestratorConfigurationError(
                "MAX_REVISE_ROUNDS must be between 1 and 3"
            )
        if not 8 <= pass_score <= 10:
            raise OrchestratorConfigurationError("PASS_SCORE must be between 8 and 10")
        self.extractor = extractor
        self.critic = critic
        self.max_revise_rounds = max_revise_rounds
        self.pass_score = pass_score
        self.evolution_logger = evolution_logger or EvolutionLogger()
        self.extractor_model = extractor_model
        self.critic_model = critic_model

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        fewshot_provider: FewShotProvider | None = None,
    ) -> NI43101Orchestrator:
        resolved = settings or get_settings()
        return cls(
            extractor=DeepSeekExtractor.from_settings(
                resolved,
                fewshot_provider=fewshot_provider,
            ),
            critic=CriticMaster.from_settings(resolved),
            max_revise_rounds=resolved.max_revise_rounds,
            pass_score=resolved.pass_score,
            extractor_model=resolved.extractor_model,
            critic_model=resolved.critic_model,
        )

    def run(
        self,
        pdf_path: str | Path,
        *,
        document_id: str | None = None,
        case_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Locate source pages and execute the bounded audit pipeline."""

        path = Path(pdf_path)
        resolved_document_id = document_id or path.stem
        self._notify(progress_callback, "正在定位候选页")
        candidate_pages = locate_candidate_pages(path, top_k=3)
        self._notify(progress_callback, "正在扩展候选页前后文")
        context_pages = get_candidate_context(
            path,
            candidate_pages,
            surrounding_pages=1,
        )
        return self.run_with_pages(
            resolved_document_id,
            candidate_pages,
            context_pages,
            case_id=case_id,
            pdf_name=path.name,
            progress_callback=progress_callback,
        )

    def run_with_pages(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage] | None = None,
        *,
        case_id: str | None = None,
        pdf_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Execute against already-located pages, primarily for deterministic tests."""

        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if not candidate_pages:
            raise ValueError("candidate_pages must not be empty")

        context = list(context_pages or ())
        resolved_case_id = case_id or uuid4().hex
        resolved_pdf_name = pdf_name or f"{document_id}.pdf"
        critic_pages = self._deduplicate_pages([*candidate_pages, *context])
        rounds: list[RoundResult] = []
        previous_extraction: ExtractionResult | None = None
        previous_validators: list[DeterministicValidationResult] = []
        previous_issues: list[CriticIssue] = []

        for round_index in range(self.max_revise_rounds):
            round_number = round_index + 1
            action = "DeepSeek 初始抽取" if round_index == 0 else "DeepSeek 修订抽取"
            self._notify(
                progress_callback,
                f"第 {round_number}/{self.max_revise_rounds} 轮：{action}",
            )
            try:
                if round_index == 0:
                    extractor_result = self.extractor.extract(
                        document_id,
                        candidate_pages,
                        context,
                        input_mode="candidate_pages_with_context",
                    )
                else:
                    if previous_extraction is None:
                        raise RuntimeError("revision requires a previous extraction")
                    extractor_result = self.extractor.revise(
                        document_id,
                        candidate_pages,
                        context,
                        previous_extraction,
                        [
                            result
                            for result in previous_validators
                            if result.status != "PASS"
                        ],
                        previous_issues,
                        input_mode="candidate_pages_with_context_revision",
                    )
            except ExtractorOutputError:
                self._record_llm_format_error(
                    case_id=resolved_case_id,
                    document_id=document_id,
                    pdf_name=resolved_pdf_name,
                    round_number=round_index + 1,
                    candidate_pages=candidate_pages,
                    stage="extractor",
                )
                raise

            self._notify(
                progress_callback,
                f"第 {round_number}/{self.max_revise_rounds} 轮："
                "单位归一化与确定性校验",
            )
            normalized_records, validator_results = self._normalize_and_validate(
                extractor_result.records
            )
            self._notify(
                progress_callback,
                f"第 {round_number}/{self.max_revise_rounds} 轮："
                "GLM CriticMaster 对抗审核",
            )
            try:
                critic_result = self.critic.review(
                    critic_pages,
                    extractor_result,
                    normalized_records,
                    validator_results,
                )
            except CriticOutputError:
                self._record_llm_format_error(
                    case_id=resolved_case_id,
                    document_id=document_id,
                    pdf_name=resolved_pdf_name,
                    round_number=round_index + 1,
                    candidate_pages=candidate_pages,
                    stage="critic",
                    extraction=extractor_result,
                    validator_results=validator_results,
                )
                raise

            if self._can_pass(
                extractor_result,
                validator_results,
                critic_result,
            ):
                rounds.append(
                    RoundResult(
                        round_number=round_index + 1,
                        extractor_result=extractor_result,
                        normalized_records=normalized_records,
                        validator_results=validator_results,
                        critic_result=critic_result,
                        decision="pass",
                    )
                )
                self._notify(
                    progress_callback,
                    f"第 {round_number} 轮完成：评分 {critic_result.score}，PASS",
                )
                return PipelineResult(
                    document_id=document_id,
                    status="pass",
                    needs_human_review=False,
                    accepted_records=normalized_records,
                    candidate_records=[],
                    rounds=rounds,
                    final_score=critic_result.score,
                    abstain_reason=None,
                )

            final_round = round_index == self.max_revise_rounds - 1
            round_result = RoundResult(
                round_number=round_index + 1,
                extractor_result=extractor_result,
                normalized_records=normalized_records,
                validator_results=validator_results,
                critic_result=critic_result,
                decision="abstain" if final_round else "revise",
            )
            rounds.append(round_result)
            self._notify(
                progress_callback,
                f"第 {round_number} 轮完成：评分 {critic_result.score}，"
                f"{'进入人工审核' if final_round else '需要修订'}",
            )
            self._record_evolution(
                case_id=resolved_case_id,
                pdf_name=resolved_pdf_name,
                round_result=round_result,
            )
            previous_extraction = extractor_result
            previous_validators = validator_results
            previous_issues = list(critic_result.issues)

            if final_round:
                return PipelineResult(
                    document_id=document_id,
                    status="abstain",
                    needs_human_review=True,
                    accepted_records=[],
                    candidate_records=list(extractor_result.records),
                    rounds=rounds,
                    final_score=critic_result.score,
                    abstain_reason=self._abstain_reason(
                        extractor_result,
                        validator_results,
                        critic_result,
                    ),
                )

        raise RuntimeError("bounded pipeline ended without a terminal decision")

    @staticmethod
    def _notify(
        callback: ProgressCallback | None,
        message: str,
    ) -> None:
        """Report best-effort progress without affecting pipeline decisions."""

        if callback is None:
            return
        try:
            callback(message)
        except Exception as error:
            warnings.warn(
                f"progress callback failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _record_evolution(
        self,
        *,
        case_id: str,
        pdf_name: str,
        round_result: RoundResult,
    ) -> None:
        """Best-effort side effect that cannot alter the pipeline decision."""

        if round_result.decision not in {"revise", "abstain"}:
            return
        try:
            record = build_round_record(
                case_id=case_id,
                document_id=round_result.extractor_result.document_id,
                pdf_name=pdf_name,
                extractor_model=self.extractor_model,
                critic_model=self.critic_model,
                round_number=round_result.round_number,
                extraction=round_result.extractor_result,
                validator_results=round_result.validator_results,
                critic_result=round_result.critic_result,
                decision=round_result.decision,
                pass_score=self.pass_score,
            )
            self.evolution_logger.append(record)
        except Exception as error:  # evolution must never block a pipeline decision
            warnings.warn(
                f"evolution logging failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _record_llm_format_error(
        self,
        *,
        case_id: str,
        document_id: str,
        pdf_name: str,
        round_number: int,
        candidate_pages: Sequence[CandidatePage],
        stage: Literal["extractor", "critic"],
        extraction: ExtractionResult | None = None,
        validator_results: Sequence[DeterministicValidationResult] = (),
    ) -> None:
        """Best-effort logging for schema-invalid model output."""

        try:
            self.evolution_logger.log_llm_format_error(
                case_id=case_id,
                document_id=document_id,
                pdf_name=pdf_name,
                extractor_model=self.extractor_model,
                critic_model=self.critic_model,
                round_number=round_number,
                candidate_pages=[page.page_number for page in candidate_pages],
                stage=stage,
                extraction=extraction,
                validator_results=validator_results,
            )
        except Exception as error:  # logging cannot replace the controlled LLM error
            warnings.warn(
                f"evolution LLM format logging failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _can_pass(
        self,
        extractor_result: ExtractionResult,
        validator_results: Sequence[DeterministicValidationResult],
        critic_result: CriticResult,
    ) -> bool:
        return (
            bool(extractor_result.records)
            and critic_result.score >= self.pass_score
            and critic_result.verdict == "pass"
            and not any(result.status == "HARD_FAIL" for result in validator_results)
            and not extractor_result.uncertain
            and not any(record.uncertain for record in extractor_result.records)
            and not any(
                issue.severity == "critical" for issue in critic_result.issues
            )
        )

    @classmethod
    def _normalize_and_validate(
        cls,
        raw_records: Sequence[RawResourceRecord],
    ) -> tuple[list[ResourceRecord], list[DeterministicValidationResult]]:
        normalized_records: list[ResourceRecord] = []
        validator_results: list[DeterministicValidationResult] = []

        for raw_index, raw_record in enumerate(raw_records):
            normalized_index: int | None = None
            try:
                grade_unit = cls._normalized_grade_unit(raw_record)
                metal_source_unit = cls._contained_metal_source_unit(raw_record)
                normalized_record = ResourceRecord(
                    document_id=raw_record.document_id,
                    location=raw_record.location,
                    category=raw_record.category,
                    commodity=raw_record.commodity,
                    tonnage_mt=normalize_ore_tonnage(
                        raw_record.tonnage_value,
                        raw_record.tonnage_unit,
                    ),
                    grade=raw_record.grade_value,
                    grade_unit=grade_unit,
                    contained_metal=normalize_contained_metal(
                        raw_record.metal_value,
                        metal_source_unit,
                        raw_record.commodity,
                    ),
                    metal_unit="oz" if raw_record.commodity == "Au" else "t",
                    source_page=raw_record.source_page,
                    table_title=raw_record.table_title,
                    evidence_text=raw_record.evidence_text,
                    source_values={
                        "tonnage_value": raw_record.tonnage_value,
                        "tonnage_unit": raw_record.tonnage_unit,
                        "grade_value": raw_record.grade_value,
                        "grade_unit": raw_record.grade_unit,
                        "metal_value": raw_record.metal_value,
                        "metal_unit": raw_record.metal_unit,
                    },
                    confidence=raw_record.confidence,
                )
                normalized_index = len(normalized_records)
                normalized_records.append(normalized_record)

                category_check = validate_category(normalized_record.category)
                commodity_check = validate_commodity(
                    normalized_record.commodity,
                    normalized_record.grade_unit,
                    normalized_record.metal_unit,
                )
                math_check = validate_metal_consistency(normalized_record)
                messages = cls._validation_messages(
                    category_check.reason,
                    commodity_check.reason,
                    math_check.status,
                    math_check.relative_error,
                )
                status = math_check.status
                if (
                    category_check.status == "HARD_FAIL"
                    or commodity_check.status == "HARD_FAIL"
                ):
                    status = "HARD_FAIL"
                validator_results.append(
                    DeterministicValidationResult(
                        record_index=raw_index,
                        normalized_record_index=normalized_index,
                        expected=math_check.expected,
                        actual=math_check.actual,
                        relative_error=math_check.relative_error,
                        status=status,
                        messages=messages,
                    )
                )
            except (TypeError, ValueError, ValidationError) as error:
                if normalized_index is not None:
                    normalized_records.pop()
                validator_results.append(
                    DeterministicValidationResult(
                        record_index=raw_index,
                        normalized_record_index=None,
                        status="HARD_FAIL",
                        messages=[f"normalization failed: {error}"],
                    )
                )

        return normalized_records, validator_results

    @staticmethod
    def _normalized_grade_unit(raw_record: RawResourceRecord) -> str:
        compact = " ".join(raw_record.grade_unit.split()).casefold()
        if raw_record.commodity == "Au" and compact in {"g/t", "g/t au"}:
            return "g/t Au"
        if raw_record.commodity == "Cu" and compact in {"%", "% cu"}:
            return "% Cu"
        raise ValueError(
            f"grade unit {raw_record.grade_unit!r} is incompatible with "
            f"{raw_record.commodity}"
        )

    @staticmethod
    def _contained_metal_source_unit(raw_record: RawResourceRecord) -> str:
        parts = raw_record.metal_unit.split()
        if parts and parts[-1].casefold() == raw_record.commodity.casefold():
            parts.pop()
        unit = " ".join(parts).strip()
        if not unit:
            raise ValueError("contained metal unit is empty after commodity suffix")
        return unit

    @staticmethod
    def _validation_messages(
        category_reason: str | None,
        commodity_reason: str | None,
        math_status: str,
        relative_error: float,
    ) -> list[str]:
        messages = [
            reason for reason in (category_reason, commodity_reason) if reason
        ]
        if math_status == "WARNING":
            messages.append(
                f"contained metal relative error {relative_error:.6f} exceeds 5%"
            )
        elif math_status == "HARD_FAIL":
            messages.append(
                f"contained metal relative error {relative_error:.6f} exceeds 10%"
            )
        return messages

    def _abstain_reason(
        self,
        extractor_result: ExtractionResult,
        validator_results: Sequence[DeterministicValidationResult],
        critic_result: CriticResult,
    ) -> str:
        reasons: list[str] = []
        if not extractor_result.records:
            reasons.append("extractor returned no target Indicated/Inferred records")
        if extractor_result.uncertain or any(
            record.uncertain for record in extractor_result.records
        ):
            details = extractor_result.uncertainty_reasons or [
                record.uncertainty_reason
                for record in extractor_result.records
                if record.uncertainty_reason
            ]
            reasons.append(
                "extractor uncertainty"
                + (f": {'; '.join(details)}" if details else "")
            )
        if critic_result.score < self.pass_score:
            reasons.append(
                f"critic score {critic_result.score} is below {self.pass_score:g}"
            )
        if critic_result.verdict != "pass":
            reasons.append(f"critic verdict is {critic_result.verdict}")

        hard_failures = [
            result for result in validator_results if result.status == "HARD_FAIL"
        ]
        if hard_failures:
            indexes = ", ".join(str(result.record_index) for result in hard_failures)
            reasons.append(f"deterministic HARD_FAIL on record(s) {indexes}")

        critical_codes = sorted(
            {issue.code for issue in critic_result.issues if issue.severity == "critical"}
        )
        if critical_codes:
            reasons.append(f"unresolved critical issue(s): {', '.join(critical_codes)}")

        return "; ".join(reasons) or "pass conditions were not satisfied"

    @staticmethod
    def _deduplicate_pages(
        pages: Sequence[CandidatePage],
    ) -> list[CandidatePage]:
        by_index: dict[int, CandidatePage] = {}
        for page in pages:
            by_index.setdefault(page.page_index, page)
        return [by_index[index] for index in sorted(by_index)]


Orchestrator = NI43101Orchestrator


def run_pipeline(
    pdf_path: str | Path,
    *,
    document_id: str | None = None,
    settings: Settings | None = None,
) -> PipelineResult:
    """Create configured agents and execute one PDF pipeline."""

    return NI43101Orchestrator.from_settings(settings).run(
        pdf_path,
        document_id=document_id,
    )
