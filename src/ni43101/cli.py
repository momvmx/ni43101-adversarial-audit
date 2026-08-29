"""Typer CLI for extraction and baseline/evolution evaluation."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path

import orjson
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ni43101.config import Settings
from ni43101.critic import CriticError
from ni43101.evaluator import (
    DEFAULT_REPORT_PATH,
    EvaluationError,
    EvaluationMetrics,
    Evaluator,
    render_metrics,
    write_evaluation_report,
)
from ni43101.evolution import DEFAULT_EVOLUTION_PATH, EvolutionLogger
from ni43101.extractor import ExtractorError
from ni43101.fewshot import EvolutionFewShotSelector, FewShotProvider
from ni43101.orchestrator import NI43101Orchestrator, PipelineResult
from ni43101.pdf_locator import (
    CandidatePage,
    get_candidate_context,
    locate_candidate_pages,
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="NI 43-101 矿产资源量抽取与对抗审核。",
)
console = Console()


class FewShotMode(str, Enum):
    OFF = "off"
    EVOLUTION = "evolution"


class CLIConfigurationError(RuntimeError):
    """Raised when safe execution configuration is incomplete."""


@app.command()
def extract(
    pdf: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="待处理的 NI 43-101 PDF。",
    ),
    fewshot: FewShotMode = typer.Option(
        FewShotMode.OFF,
        "--fewshot",
        case_sensitive=False,
        help="关闭回放，或使用防泄漏 Evolution 回放。",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="PipelineResult JSON 路径，默认写入 outputs/runs。",
    ),
    evolution_log: Path = typer.Option(
        DEFAULT_EVOLUTION_PATH,
        "--evolution-log",
        help="Append-only Evolution JSONL 路径。",
    ),
    debug: bool = typer.Option(False, "--debug", help="显示安全审核详情。"),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="显示 PDF、LLM、校验与审核阶段进度。",
    ),
) -> None:
    """抽取并对抗审核一份 PDF。"""

    settings = Settings()
    try:
        _require_runtime_credentials(settings)
        provider = _fewshot_provider(fewshot, evolution_log)
        result, candidates = _run_pdf(
            pdf,
            settings=settings,
            fewshot_provider=provider,
            evolution_log=evolution_log,
            show_progress=progress,
        )
        output_path = output or Path("outputs/runs") / f"{pdf.stem}.json"
        _write_pipeline_result(result, output_path)
    except (CLIConfigurationError, ExtractorError, CriticError, OSError, ValueError) as error:
        _fail(str(error))

    _render_extraction_summary(result)
    console.print(f"结果文件：{output_path}")
    if debug:
        _render_debug(candidates, result)


@app.command("evaluate")
def evaluate_command(
    pdf_dir: Path = typer.Option(
        Path("data/pdfs"),
        "--pdf-dir",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    gt_dir: Path = typer.Option(
        Path("data/ground_truth"),
        "--gt-dir",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    fewshot: FewShotMode = typer.Option(
        FewShotMode.OFF,
        "--fewshot",
        case_sensitive=False,
    ),
    output: Path = typer.Option(DEFAULT_REPORT_PATH, "--output"),
    evolution_log: Path = typer.Option(
        DEFAULT_EVOLUTION_PATH,
        "--evolution-log",
    ),
    debug: bool = typer.Option(False, "--debug"),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="显示文档与每轮 Pipeline 进度。",
    ),
) -> None:
    """运行基线评估或基线与 Evolution 对比评估。"""

    settings = Settings()
    try:
        _require_runtime_credentials(settings)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise CLIConfigurationError(f"目录中没有 PDF 文件：{pdf_dir}")

        logger = EvolutionLogger(evolution_log)
        # Freeze before any extraction/evaluation writes from this invocation.
        frozen_history = logger.read()
        evaluator = Evaluator(
            evolution_logger=logger,
            extractor_model=settings.extractor_model,
            critic_model=settings.critic_model,
            field_tolerance=settings.field_tolerance,
        )
        truths = evaluator.load_ground_truth_directory(gt_dir)
        if not truths:
            raise EvaluationError(f"目录中没有 Ground Truth JSON 文件：{gt_dir}")
        _validate_pdf_truth_coverage(pdfs, set(truths))

        baseline_results = _run_pdf_set(
            pdfs,
            settings=settings,
            fewshot_provider=None,
            evolution_log=evolution_log,
            mode_name="baseline",
            debug=debug,
            show_progress=progress,
        )
        baseline_metrics, _ = evaluator.evaluate(baseline_results, truths)

        if fewshot == FewShotMode.OFF:
            write_evaluation_report(baseline_metrics, output)
            _render_named_metrics("基线模式", baseline_metrics)
        else:
            provider = EvolutionFewShotSelector(records=frozen_history)
            evolution_results = _run_pdf_set(
                pdfs,
                settings=settings,
                fewshot_provider=provider,
                evolution_log=evolution_log,
                mode_name="evolution",
                debug=debug,
                show_progress=progress,
            )
            evolution_metrics, _ = evaluator.evaluate(evolution_results, truths)
            _write_comparison_report(
                baseline_metrics,
                evolution_metrics,
                output,
            )
            _render_comparison(baseline_metrics, evolution_metrics)
    except (
        CLIConfigurationError,
        EvaluationError,
        ExtractorError,
        CriticError,
        OSError,
        ValueError,
    ) as error:
        _fail(str(error))

    console.print(f"评估报告：{output}")


def _run_pdf(
    pdf: Path,
    *,
    settings: Settings,
    fewshot_provider: FewShotProvider | None,
    evolution_log: Path,
    show_progress: bool = True,
    progress_label: str | None = None,
) -> tuple[PipelineResult, list[CandidatePage]]:
    label = progress_label or pdf.name

    def execute(report: Callable[[str], None] | None) -> tuple[PipelineResult, list[CandidatePage]]:
        if report is not None:
            report("正在定位候选页")
        candidates = locate_candidate_pages(pdf, top_k=3)
        if report is not None:
            pages = ", ".join(str(page.page_number) for page in candidates)
            report(f"正在扩展候选页上下文（页码：{pages}）")
        context = get_candidate_context(pdf, candidates, surrounding_pages=1)
        orchestrator = NI43101Orchestrator.from_settings(
            settings,
            fewshot_provider=fewshot_provider,
        )
        orchestrator.evolution_logger = EvolutionLogger(evolution_log)
        result = orchestrator.run_with_pages(
            pdf.stem,
            candidates,
            context,
            pdf_name=pdf.name,
            progress_callback=report,
        )
        return result, candidates

    if not show_progress:
        return execute(None)

    progress_display = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress_display:
        task_id = progress_display.add_task(label, total=None)

        def report(message: str) -> None:
            progress_display.update(
                task_id,
                description=f"{label} · {message}",
            )

        return execute(report)


def _run_pdf_set(
    pdfs: list[Path],
    *,
    settings: Settings,
    fewshot_provider: FewShotProvider | None,
    evolution_log: Path,
    mode_name: str,
    debug: bool,
    show_progress: bool,
) -> list[PipelineResult]:
    results: list[PipelineResult] = []
    mode_labels = {"baseline": "基线", "evolution": "Evolution"}
    total_documents = len(pdfs)
    for document_index, pdf in enumerate(pdfs, start=1):
        mode_label = mode_labels.get(mode_name, mode_name)
        result, candidates = _run_pdf(
            pdf,
            settings=settings,
            fewshot_provider=fewshot_provider,
            evolution_log=evolution_log,
            show_progress=show_progress,
            progress_label=(
                f"[{mode_label} {document_index}/{total_documents}] {pdf.name}"
            ),
        )
        destination = Path("outputs/runs") / mode_name / f"{pdf.stem}.json"
        _write_pipeline_result(result, destination)
        results.append(result)
        if debug:
            _render_debug(candidates, result)
    return results


def _fewshot_provider(
    mode: FewShotMode,
    evolution_log: Path,
) -> FewShotProvider | None:
    if mode == FewShotMode.OFF:
        return None
    return EvolutionFewShotSelector(evolution_log)


def _require_runtime_credentials(settings: Settings) -> None:
    missing: list[str] = []
    if settings.extractor_api_key is None:
        missing.append("EXTRACTOR_API_KEY")
    if settings.critic_api_key is None:
        missing.append("CRITIC_API_KEY")
    if not settings.critic_base_url:
        missing.append("CRITIC_BASE_URL")
    if missing:
        raise CLIConfigurationError(
            "缺少必需环境变量：" + ", ".join(missing)
        )


def _validate_pdf_truth_coverage(pdfs: list[Path], document_ids: set[str]) -> None:
    pdf_document_ids = {pdf.stem for pdf in pdfs}
    missing_truth = sorted(pdf_document_ids - document_ids)
    if missing_truth:
        raise EvaluationError(
            "以下 PDF 缺少 Ground Truth："
            + ", ".join(missing_truth)
        )
    missing_pdfs = sorted(document_ids - pdf_document_ids)
    if missing_pdfs:
        raise EvaluationError(
            "以下 Ground Truth 缺少对应 PDF：" + ", ".join(missing_pdfs)
        )


def _write_pipeline_result(result: PipelineResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        + b"\n"
    )


def _write_comparison_report(
    baseline: EvaluationMetrics,
    evolution: EvaluationMetrics,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline.model_dump(mode="json"),
        "evolution": evolution.model_dump(mode="json"),
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")


def _render_extraction_summary(result: PipelineResult) -> None:
    table = Table(title="NI 43-101 矿产资源量抽取结果")
    table.add_column("字段")
    table.add_column("值")
    candidate_pages = (
        result.rounds[0].extractor_result.candidate_pages if result.rounds else []
    )
    status = "通过（PASS）" if result.status == "pass" else "弃权（ABSTAIN）"
    table.add_row("文档", result.document_id)
    table.add_row("候选页", ", ".join(map(str, candidate_pages)) or "无")
    table.add_row("执行轮数", str(len(result.rounds)))
    table.add_row("最终 Critic 评分", str(result.final_score))
    table.add_row("状态", status)
    table.add_row("接受记录数", str(len(result.accepted_records)))
    table.add_row("弃权原因", result.abstain_reason or "-")
    console.print(table)
    if result.accepted_records:
        records = Table(title="已接受的矿产资源量记录")
        records.add_column("地点")
        records.add_column("类别")
        records.add_column("商品")
        records.add_column("矿石量/Mt", justify="right")
        records.add_column("品位", justify="right")
        records.add_column("金属量", justify="right")
        records.add_column("页码", justify="right")
        category_labels = {
            "Indicated": "指示",
            "Inferred": "推断",
        }
        commodity_labels = {"Au": "金 (Au)", "Cu": "铜 (Cu)"}
        for record in result.accepted_records:
            records.add_row(
                record.location,
                category_labels[record.category],
                commodity_labels[record.commodity],
                _format_number(record.tonnage_mt),
                f"{_format_number(record.grade)} {record.grade_unit}",
                f"{_format_number(record.contained_metal)} {record.metal_unit}",
                str(record.source_page),
            )
        console.print(records)


def _format_number(value: float) -> str:
    """Format normalized values without binary tails or scientific notation."""

    return f"{value:,.6f}".rstrip("0").rstrip(".")


def _render_debug(candidates: list[CandidatePage], result: PipelineResult) -> None:
    candidate_table = Table(title="候选页调试信息")
    candidate_table.add_column("页码", justify="right")
    candidate_table.add_column("评分", justify="right")
    candidate_table.add_column("正向关键词")
    candidate_table.add_column("负向关键词")
    candidate_table.add_column("评分原因")
    for page in candidates:
        candidate_table.add_row(
            str(page.page_number),
            f"{page.score:.1f}",
            ", ".join(page.matched_positive_keywords),
            ", ".join(page.matched_negative_keywords),
            "; ".join(page.reasons),
        )
    console.print(candidate_table)

    for round_result in result.rounds:
        decision_labels = {"pass": "通过", "revise": "修订", "abstain": "弃权"}
        console.rule(
            f"第 {round_result.round_number} 轮："
            f"{decision_labels[round_result.decision]}"
        )
        console.print("原始抽取结果")
        console.print_json(data=round_result.extractor_result.model_dump(mode="json"))
        console.print("归一化记录")
        console.print_json(
            data=[
                record.model_dump(mode="json")
                for record in round_result.normalized_records
            ]
        )
        console.print("确定性校验结果")
        console.print_json(
            data=[
                validation.model_dump(mode="json")
                for validation in round_result.validator_results
            ]
        )
        console.print("Critic 问题")
        console.print_json(
            data=[issue.model_dump(mode="json") for issue in round_result.critic_result.issues]
        )
        if round_result.decision == "revise":
            console.print(f"修订原因：{_revision_reason(round_result)}")


def _revision_reason(round_result) -> str:
    reasons = [
        message
        for validation in round_result.validator_results
        if validation.status != "PASS"
        for message in validation.messages
    ]
    reasons.extend(issue.message for issue in round_result.critic_result.issues)
    if round_result.critic_result.score < 8:
        reasons.append(
            f"Critic 评分 {round_result.critic_result.score} 低于通过阈值"
        )
    return "; ".join(reasons) or round_result.critic_result.summary


def _render_named_metrics(name: str, metrics: EvaluationMetrics) -> None:
    console.print(f"[bold]{name}[/bold]")
    render_metrics(metrics, console)


def _render_comparison(
    baseline: EvaluationMetrics,
    evolution: EvaluationMetrics,
) -> None:
    table = Table(title="基线模式与 Evolution 模式对比")
    table.add_column("指标")
    table.add_column("值", justify="right")
    table.add_row("基线字段准确率", f"{baseline.field_accuracy:.2%}")
    table.add_row("Evolution 字段准确率", f"{evolution.field_accuracy:.2%}")
    table.add_row("基线弃权率", f"{baseline.abstain_rate:.2%}")
    table.add_row("Evolution 弃权率", f"{evolution.abstain_rate:.2%}")
    table.add_row(
        "基线不安全接受率",
        f"{baseline.unsafe_accept_rate:.2%}",
    )
    table.add_row(
        "Evolution 不安全接受率",
        f"{evolution.unsafe_accept_rate:.2%}",
    )
    console.print(table)


def _fail(message: str) -> None:
    console.print(f"[red]错误：[/red] {message}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
