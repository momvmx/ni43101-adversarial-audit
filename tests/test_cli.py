from pathlib import Path

import orjson
import pytest
from typer.testing import CliRunner

from ni43101.cli import (
    CLIConfigurationError,
    _render_extraction_summary,
    _require_runtime_credentials,
    _validate_pdf_truth_coverage,
    _write_comparison_report,
    app,
    console,
)
from ni43101.config import Settings
from ni43101.evaluator import EvaluationError, EvaluationMetrics
from ni43101.orchestrator import PipelineResult


def _metrics(field_accuracy: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        documents_total=2,
        documents_passed=1,
        documents_abstained=1,
        abstain_rate=0.5,
        total_numeric_fields=6,
        correct_numeric_fields=round(field_accuracy * 6),
        field_accuracy=field_accuracy,
        accepted_records=1,
        accepted_record_accuracy=field_accuracy,
        unsafe_accept_count=1,
        unsafe_accept_rate=1,
        safe_abstain_count=1,
    )


def test_missing_credentials_names_are_reported_without_values() -> None:
    settings = Settings(
        extractor_api_key=None,
        critic_api_key=None,
        critic_base_url=None,
    )

    try:
        _require_runtime_credentials(settings)
    except CLIConfigurationError as error:
        message = str(error)
    else:
        raise AssertionError("missing credentials should fail")

    assert "EXTRACTOR_API_KEY" in message
    assert "CRITIC_API_KEY" in message
    assert "CRITIC_BASE_URL" in message
    assert "sk-" not in message


def test_extract_command_stops_before_pdf_processing_without_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("EXTRACTOR_API_KEY", "CRITIC_API_KEY", "CRITIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    pdf = tmp_path / "mock.pdf"
    pdf.write_bytes(b"not processed")

    result = CliRunner().invoke(app, ["extract", str(pdf)])

    assert result.exit_code == 1
    assert "EXTRACTOR_API_KEY" in result.output
    assert "CRITIC_API_KEY" in result.output
    assert "CRITIC_BASE_URL" in result.output
    assert "错误" in result.output
    assert "缺少必需环境变量" in result.output


def test_comparison_report_has_baseline_and_evolution(tmp_path: Path) -> None:
    output = tmp_path / "evaluation_report.json"

    _write_comparison_report(_metrics(0.5), _metrics(1.0), output)

    payload = orjson.loads(output.read_bytes())
    assert payload["baseline"]["field_accuracy"] == 0.5
    assert payload["evolution"]["field_accuracy"] == 1.0


def test_cli_exposes_extract_and_evaluate_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "extract" in result.output
    assert "evaluate" in result.output


def test_evaluation_requires_ground_truth_for_every_pdf() -> None:
    with pytest.raises(EvaluationError, match="缺少 Ground Truth.*doc-b"):
        _validate_pdf_truth_coverage(
            [Path("doc-a.pdf"), Path("doc-b.pdf")],
            {"doc-a"},
        )


def test_evaluation_requires_pdf_for_every_ground_truth() -> None:
    with pytest.raises(EvaluationError, match="缺少对应 PDF.*doc-b"):
        _validate_pdf_truth_coverage(
            [Path("doc-a.pdf")],
            {"doc-a", "doc-b"},
        )


def test_extraction_summary_uses_chinese_labels() -> None:
    result = PipelineResult(
        document_id="report-1",
        status="abstain",
        needs_human_review=True,
        accepted_records=[],
        candidate_records=[],
        rounds=[],
        final_score=7,
        abstain_reason="需要人工审核",
    )

    with console.capture() as capture:
        _render_extraction_summary(result)

    output = capture.get()
    assert "矿产资源量抽取结果" in output
    assert "候选页" in output
    assert "弃权（ABSTAIN）" in output
    assert "需要人工审核" in output
