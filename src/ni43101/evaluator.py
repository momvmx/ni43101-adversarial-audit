"""Offline evaluator; this is the only module allowed to read Ground Truth."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import orjson
import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich.console import Console
from rich.table import Table

from ni43101.evolution import EvolutionLogger, FailureType
from ni43101.orchestrator import PipelineResult
from ni43101.schemas import Commodity, ResourceCategory, ResourceRecord


FIELD_TOLERANCE = 0.05
CORE_NUMERIC_FIELDS = ("tonnage_mt", "grade", "contained_metal")
DEFAULT_REPORT_PATH = Path("outputs/evaluation_report.json")


class _EvaluationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GroundTruthRecord(_EvaluationSchema):
    location: str = Field(min_length=1)
    category: ResourceCategory
    commodity: Commodity
    tonnage_mt: float
    grade: float
    grade_unit: Literal["g/t Au", "% Cu"]
    contained_metal: float
    metal_unit: Literal["oz", "t"]

    @model_validator(mode="after")
    def validate_semantics(self) -> GroundTruthRecord:
        for field_name in CORE_NUMERIC_FIELDS:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        expected_units = (
            ("g/t Au", "oz") if self.commodity == "Au" else ("% Cu", "t")
        )
        if (self.grade_unit, self.metal_unit) != expected_units:
            raise ValueError(
                f"ground truth units are incompatible with {self.commodity}"
            )
        return self


class GroundTruth(_EvaluationSchema):
    document_id: str = Field(min_length=1)
    records: list[GroundTruthRecord]

    @model_validator(mode="after")
    def validate_unique_keys(self) -> GroundTruth:
        keys = [record_key(record) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "ground truth contains duplicate location/category/commodity keys"
            )
        return self


class NumericFieldEvaluation(_EvaluationSchema):
    record_key: str
    field: Literal["tonnage_mt", "grade", "contained_metal"]
    prediction: float | None
    truth: float
    relative_error: float | None
    correct: bool


class GroundTruthDiff(_EvaluationSchema):
    numeric_errors: list[NumericFieldEvaluation]
    missing_record_keys: list[str]
    unexpected_record_keys: list[str]


class DocumentEvaluation(_EvaluationSchema):
    document_id: str
    pipeline_status: Literal["pass", "abstain"]
    numeric_fields_total: int
    numeric_fields_correct: int
    accepted_records: int
    correct_accepted_records: int
    unsafe_accept: bool
    safe_abstain: bool
    extraction_correct: bool
    ground_truth_diff: GroundTruthDiff


class EvaluationMetrics(_EvaluationSchema):
    documents_total: int = Field(ge=0)
    documents_passed: int = Field(ge=0)
    documents_abstained: int = Field(ge=0)
    abstain_rate: float = Field(ge=0, le=1)
    total_numeric_fields: int = Field(ge=0)
    correct_numeric_fields: int = Field(ge=0)
    field_accuracy: float = Field(ge=0, le=1)
    accepted_records: int = Field(ge=0)
    accepted_record_accuracy: float = Field(ge=0, le=1)
    unsafe_accept_count: int = Field(ge=0)
    unsafe_accept_rate: float = Field(ge=0, le=1)
    safe_abstain_count: int = Field(ge=0)


class EvaluationError(RuntimeError):
    """Controlled error for malformed or mismatched offline evaluation input."""


def normalize_location(location: str) -> str:
    """Normalize case, whitespace, Unicode, and simple location punctuation."""

    normalized = unicodedata.normalize("NFKC", location).strip().casefold()
    normalized = normalized.replace("&", " and ")
    normalized = "".join(
        " "
        if unicodedata.category(character).startswith(("P", "S"))
        else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def record_key(record: GroundTruthRecord | ResourceRecord) -> tuple[str, str, str]:
    """Match location flexibly but category and commodity exactly."""

    return (
        normalize_location(record.location),
        record.category,
        record.commodity,
    )


def display_record_key(key: tuple[str, str, str]) -> str:
    return " | ".join(key)


def relative_error(prediction: float, truth: float) -> float:
    """Return absolute relative error, with explicit behavior for a zero truth."""

    prediction_value = float(prediction)
    truth_value = float(truth)
    if not math.isfinite(prediction_value) or not math.isfinite(truth_value):
        return math.inf
    if truth_value == 0:
        return 0.0 if prediction_value == 0 else math.inf
    return abs(prediction_value - truth_value) / abs(truth_value)


def field_is_correct(
    prediction: float,
    truth: float,
    *,
    tolerance: float = FIELD_TOLERANCE,
) -> bool:
    return relative_error(prediction, truth) <= tolerance


def evaluate_document(
    pipeline_result: PipelineResult,
    ground_truth: GroundTruth,
    *,
    tolerance: float = FIELD_TOLERANCE,
) -> DocumentEvaluation:
    """Compare one pipeline result without reading files or producing side effects."""

    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be between 0 and 1")

    if pipeline_result.document_id != ground_truth.document_id:
        raise EvaluationError(
            "pipeline and ground truth document_id values do not match: "
            f"{pipeline_result.document_id!r} != {ground_truth.document_id!r}"
        )

    truth_by_key = {record_key(record): record for record in ground_truth.records}
    predictions = list(pipeline_result.accepted_records)
    prediction_by_key: dict[tuple[str, str, str], ResourceRecord] = {}
    duplicate_prediction_keys: list[tuple[str, str, str]] = []
    for prediction in predictions:
        key = record_key(prediction)
        if key in prediction_by_key:
            duplicate_prediction_keys.append(key)
            continue
        prediction_by_key[key] = prediction

    numeric_evaluations: list[NumericFieldEvaluation] = []
    correct_accepted_records = 0
    for key, truth_record in truth_by_key.items():
        prediction = prediction_by_key.get(key)
        record_correct = prediction is not None
        for field in CORE_NUMERIC_FIELDS:
            truth_value = float(getattr(truth_record, field))
            prediction_value = (
                None if prediction is None else float(getattr(prediction, field))
            )
            comparison_error = (
                None
                if prediction_value is None
                else relative_error(prediction_value, truth_value)
            )
            correct = (
                comparison_error is not None
                and comparison_error <= tolerance
            )
            record_correct = record_correct and correct
            if not correct:
                numeric_evaluations.append(
                    NumericFieldEvaluation(
                        record_key=display_record_key(key),
                        field=field,
                        prediction=prediction_value,
                        truth=truth_value,
                        relative_error=(
                            comparison_error
                            if comparison_error is not None
                            and math.isfinite(comparison_error)
                            else None
                        ),
                        correct=False,
                    )
                )
        if record_correct:
            correct_accepted_records += 1

    truth_keys = set(truth_by_key)
    prediction_keys = set(prediction_by_key)
    missing_keys = sorted(truth_keys - prediction_keys)
    unexpected_keys = sorted(prediction_keys - truth_keys)
    unexpected_keys.extend(duplicate_prediction_keys)
    total_numeric_fields = len(ground_truth.records) * len(CORE_NUMERIC_FIELDS)
    correct_numeric_fields = total_numeric_fields - len(numeric_evaluations)
    structure_correct = not missing_keys and not unexpected_keys
    extraction_correct = (
        pipeline_result.status == "pass"
        and structure_correct
        and correct_numeric_fields == total_numeric_fields
    )

    # Missing and unexpected accepted records are unsafe too: they represent
    # unavailable core comparisons and must not create a safety loophole.
    unsafe_accept = pipeline_result.status == "pass" and not extraction_correct
    return DocumentEvaluation(
        document_id=pipeline_result.document_id,
        pipeline_status=pipeline_result.status,
        numeric_fields_total=total_numeric_fields,
        numeric_fields_correct=correct_numeric_fields,
        accepted_records=len(predictions),
        correct_accepted_records=correct_accepted_records,
        unsafe_accept=unsafe_accept,
        safe_abstain=pipeline_result.status == "abstain",
        extraction_correct=extraction_correct,
        ground_truth_diff=GroundTruthDiff(
            numeric_errors=numeric_evaluations,
            missing_record_keys=[display_record_key(key) for key in missing_keys],
            unexpected_record_keys=[
                display_record_key(key) for key in unexpected_keys
            ],
        ),
    )


def aggregate_metrics(
    evaluations: Iterable[DocumentEvaluation],
) -> EvaluationMetrics:
    documents = list(evaluations)
    documents_total = len(documents)
    documents_passed = sum(item.pipeline_status == "pass" for item in documents)
    documents_abstained = sum(
        item.pipeline_status == "abstain" for item in documents
    )
    total_numeric_fields = sum(item.numeric_fields_total for item in documents)
    correct_numeric_fields = sum(item.numeric_fields_correct for item in documents)
    accepted_records = sum(item.accepted_records for item in documents)
    correct_accepted_records = sum(
        item.correct_accepted_records for item in documents
    )
    unsafe_accept_count = sum(item.unsafe_accept for item in documents)
    safe_abstain_count = sum(item.safe_abstain for item in documents)

    return EvaluationMetrics(
        documents_total=documents_total,
        documents_passed=documents_passed,
        documents_abstained=documents_abstained,
        abstain_rate=_safe_ratio(documents_abstained, documents_total),
        total_numeric_fields=total_numeric_fields,
        correct_numeric_fields=correct_numeric_fields,
        field_accuracy=_safe_ratio(correct_numeric_fields, total_numeric_fields),
        accepted_records=accepted_records,
        accepted_record_accuracy=_safe_ratio(
            correct_accepted_records,
            accepted_records,
        ),
        unsafe_accept_count=unsafe_accept_count,
        unsafe_accept_rate=_safe_ratio(unsafe_accept_count, documents_passed),
        safe_abstain_count=safe_abstain_count,
    )


class Evaluator:
    """Offline batch evaluator and the sole Ground Truth file reader."""

    def __init__(
        self,
        *,
        evolution_logger: EvolutionLogger | None = None,
        extractor_model: str = "deepseek-chat",
        critic_model: str = "GLM-4.7-Flash",
        field_tolerance: float = FIELD_TOLERANCE,
    ) -> None:
        if not 0 <= field_tolerance <= 1:
            raise ValueError("field_tolerance must be between 0 and 1")
        self.evolution_logger = evolution_logger or EvolutionLogger()
        self.extractor_model = extractor_model
        self.critic_model = critic_model
        self.field_tolerance = field_tolerance

    def load_ground_truth(self, path: str | Path) -> GroundTruth:
        """Read one Ground Truth JSON file. No online component calls this method."""

        source = Path(path)
        try:
            return GroundTruth.model_validate_json(source.read_bytes())
        except (OSError, ValidationError) as error:
            raise EvaluationError(f"invalid ground truth file {source}: {error}") from error

    def evaluate(
        self,
        pipeline_results: Sequence[PipelineResult],
        ground_truth_by_document: Mapping[str, GroundTruth],
    ) -> tuple[EvaluationMetrics, list[DocumentEvaluation]]:
        details: list[DocumentEvaluation] = []
        for pipeline_result in pipeline_results:
            try:
                ground_truth = ground_truth_by_document[pipeline_result.document_id]
            except KeyError as error:
                raise EvaluationError(
                    f"missing ground truth for {pipeline_result.document_id}"
                ) from error
            detail = evaluate_document(
                pipeline_result,
                ground_truth,
                tolerance=self.field_tolerance,
            )
            details.append(detail)
            if not detail.extraction_correct:
                self._record_evaluation_failure(pipeline_result, detail)
        return aggregate_metrics(details), details

    def load_ground_truth_directory(
        self,
        directory: str | Path,
    ) -> dict[str, GroundTruth]:
        root = Path(directory)
        truths: dict[str, GroundTruth] = {}
        for path in sorted(root.glob("*.json")):
            truth = self.load_ground_truth(path)
            if truth.document_id in truths:
                raise EvaluationError(
                    f"duplicate ground truth document_id {truth.document_id!r}"
                )
            truths[truth.document_id] = truth
        return truths

    def _record_evaluation_failure(
        self,
        pipeline_result: PipelineResult,
        detail: DocumentEvaluation,
    ) -> None:
        last_round = pipeline_result.rounds[-1] if pipeline_result.rounds else None
        failure_types = _evaluation_failure_types(detail)
        self.evolution_logger.log_evaluation_failure(
            case_id=f"evaluation:{pipeline_result.document_id}",
            document_id=pipeline_result.document_id,
            pdf_name=f"{pipeline_result.document_id}.pdf",
            extractor_model=self.extractor_model,
            critic_model=self.critic_model,
            round_number=last_round.round_number if last_round else 0,
            candidate_pages=(
                last_round.extractor_result.candidate_pages if last_round else []
            ),
            failure_types=failure_types,
            ground_truth_diff=detail.ground_truth_diff.model_dump(mode="json"),
            extractor_output=(
                last_round.extractor_result.model_dump(mode="json")
                if last_round
                else None
            ),
            validator_output=(
                [item.model_dump(mode="json") for item in last_round.validator_results]
                if last_round
                else []
            ),
            critic_output=(
                last_round.critic_result.model_dump(mode="json")
                if last_round
                else None
            ),
        )


def write_evaluation_report(
    metrics: EvaluationMetrics,
    path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        orjson.dumps(metrics.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        + b"\n"
    )
    return destination


def load_pipeline_results(path: str | Path) -> list[PipelineResult]:
    source = Path(path)
    try:
        payload = orjson.loads(source.read_bytes())
        if isinstance(payload, dict) and "results" in payload:
            payload = payload["results"]
        if isinstance(payload, list):
            return [PipelineResult.model_validate(item) for item in payload]
        return [PipelineResult.model_validate(payload)]
    except (OSError, orjson.JSONDecodeError, ValidationError, TypeError) as error:
        raise EvaluationError(f"invalid pipeline result file {source}: {error}") from error


def render_metrics(metrics: EvaluationMetrics, console: Console | None = None) -> None:
    output = console or Console()
    table = Table(title="NI 43-101 评估结果")
    table.add_column("指标")
    table.add_column("值", justify="right")
    table.add_row("字段准确率", f"{metrics.field_accuracy:.2%}")
    table.add_row("弃权率", f"{metrics.abstain_rate:.2%}")
    table.add_row("不安全接受率", f"{metrics.unsafe_accept_rate:.2%}")
    output.print(table)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    pipeline_files: list[Path] = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="一个或多个 PipelineResult JSON 文件。",
    ),
    ground_truth_dir: Path = typer.Option(
        Path("data/ground_truth"),
        "--ground-truth-dir",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        DEFAULT_REPORT_PATH,
        "--output",
        help="评估报告 JSON 输出路径。",
    ),
) -> None:
    """使用隔离的 Ground Truth 评估 PipelineResult JSON。"""

    evaluator = Evaluator()
    try:
        pipeline_results = [
            result
            for path in pipeline_files
            for result in load_pipeline_results(path)
        ]
        truths = evaluator.load_ground_truth_directory(ground_truth_dir)
        metrics, _ = evaluator.evaluate(pipeline_results, truths)
        report_path = write_evaluation_report(metrics, output)
    except EvaluationError as error:
        raise typer.BadParameter(str(error)) from error

    render_metrics(metrics)
    typer.echo(f"评估报告：{report_path}")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _evaluation_failure_types(
    detail: DocumentEvaluation,
) -> list[FailureType]:
    failures: list[FailureType] = []
    if detail.ground_truth_diff.missing_record_keys:
        failures.append(FailureType.MISSING_CATEGORY)
    if detail.ground_truth_diff.unexpected_record_keys:
        failures.append(FailureType.AMBIGUOUS_TABLE)
    numeric_fields = {
        error.field for error in detail.ground_truth_diff.numeric_errors
    }
    if "tonnage_mt" in numeric_fields:
        failures.append(FailureType.TONNAGE_UNIT_ERROR)
    if "contained_metal" in numeric_fields:
        failures.append(FailureType.METAL_UNIT_ERROR)
    if numeric_fields:
        failures.append(FailureType.MATH_INCONSISTENCY)
    if not failures:
        failures.append(FailureType.AMBIGUOUS_TABLE)
    return failures


if __name__ == "__main__":
    app()
