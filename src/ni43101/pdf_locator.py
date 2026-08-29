"""Locate pages likely to contain NI 43-101 Mineral Resource tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pymupdf
import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table


class CandidatePage(BaseModel):
    """A scored PDF page and the evidence used to rank it."""

    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    score: float
    matched_positive_keywords: list[str]
    matched_negative_keywords: list[str]
    reasons: list[str]
    text: str


@dataclass(frozen=True)
class _PageEvaluation:
    score: float
    positive_keywords: list[str]
    negative_keywords: list[str]
    reasons: list[str]


_STRONG_POSITIVE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("Mineral Resource Statement", 30.0),
    ("Mineral Resources Statement", 30.0),
    ("Mineral Resources Summary", 28.0),
)

_POSITIVE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("Mineral Resource", 8.0),
    ("Indicated", 7.0),
    ("Inferred", 8.0),
    ("Tonnage", 4.0),
    ("Tonnes", 4.0),
    ("Grade", 4.0),
    ("Contained", 5.0),
    ("Au", 2.0),
    ("Cu", 2.0),
)

_NEGATIVE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("Mineral Reserve Statement", -35.0),
    ("Mineral Reserves", -22.0),
    ("Proven", -12.0),
    ("Probable", -12.0),
)


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Extract full text from every page of a PDF in document order."""

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")

    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"PDF 受密码保护：{path}")
        return [page.get_text("text") or "" for page in document]


def _contains(text: str, keyword: str) -> bool:
    """Match a keyword case-insensitively with flexible whitespace."""

    words = re.split(r"\s+", keyword.strip())
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in words) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _evaluate_page(text: str) -> _PageEvaluation:
    score = 0.0
    positive: list[str] = []
    negative: list[str] = []
    reasons: list[str] = []

    for keyword, weight in _STRONG_POSITIVE_WEIGHTS:
        if _contains(text, keyword):
            score += weight
            positive.append(keyword)
            reasons.append(f"强 Resource 标题：{keyword}（+{weight:g}）")

    for keyword, weight in _POSITIVE_WEIGHTS:
        if _contains(text, keyword):
            score += weight
            positive.append(keyword)

    for keyword, weight in _NEGATIVE_WEIGHTS:
        if _contains(text, keyword):
            score += weight
            negative.append(keyword)
            reasons.append(f"Reserve 证据：{keyword}（{weight:g}）")

    has_indicated = "Indicated" in positive
    has_inferred = "Inferred" in positive
    has_tonnage = "Tonnes" in positive or "Tonnage" in positive
    has_grade = "Grade" in positive
    has_contained = "Contained" in positive
    has_resource = "Mineral Resource" in positive or any(
        keyword in positive for keyword, _ in _STRONG_POSITIVE_WEIGHTS
    )
    has_proven = "Proven" in negative
    has_probable = "Probable" in negative

    if has_indicated and has_inferred:
        score += 25.0
        reasons.append("同时包含 Indicated 和 Inferred（+25）")

    if has_indicated and has_inferred and has_tonnage and has_grade and has_contained:
        score += 25.0
        reasons.append(
            "资源量表特征：Indicated + Inferred + "
            "Tonnes/Tonnage + Grade + Contained（+25）"
        )

    if has_resource and has_indicated and has_inferred:
        score += 12.0
        reasons.append("Resource 术语与两个目标类别同时出现（+12）")

    if has_proven and has_probable:
        score -= 35.0
        reasons.append("储量表特征：Proven + Probable（-35）")

    if _contains(text, "Measured") and has_indicated:
        score += 2.0
        positive.append("Measured")
        reasons.append("Measured 与 Indicated 同时出现（+2）")

    if not reasons and positive:
        reasons.append("匹配到独立资源量表关键词")
    elif not positive and not negative:
        reasons.append("未匹配到 Resource 或 Reserve 证据")

    return _PageEvaluation(
        score=score,
        positive_keywords=positive,
        negative_keywords=negative,
        reasons=reasons,
    )


def score_page(text: str) -> float:
    """Return the resource-table likelihood score for one full page of text."""

    return _evaluate_page(text).score


def _make_candidate(page_index: int, text: str) -> CandidatePage:
    evaluation = _evaluate_page(text)
    return CandidatePage(
        page_index=page_index,
        page_number=page_index + 1,
        score=evaluation.score,
        matched_positive_keywords=evaluation.positive_keywords,
        matched_negative_keywords=evaluation.negative_keywords,
        reasons=evaluation.reasons,
        text=text,
    )


def locate_candidate_pages(
    pdf_path: str | Path,
    top_k: int = 5,
) -> list[CandidatePage]:
    """Return the highest-scoring pages likely to contain resource tables."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    candidates = [
        _make_candidate(page_index, text)
        for page_index, text in enumerate(extract_pages(pdf_path))
    ]
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.page_index))
    return candidates[:top_k]


def get_candidate_context(
    pdf_path: str | Path,
    candidate_pages: Sequence[CandidatePage],
    surrounding_pages: int = 1,
) -> list[CandidatePage]:
    """Return candidates plus nearby pages, deduplicated in document order."""

    if surrounding_pages < 0:
        raise ValueError("surrounding_pages cannot be negative")
    if not candidate_pages:
        return []

    pages = extract_pages(pdf_path)
    page_count = len(pages)
    context_indexes: set[int] = set()

    for candidate in candidate_pages:
        if candidate.page_index >= page_count:
            raise ValueError(
                f"candidate page index {candidate.page_index} is outside the PDF"
            )
        start = max(0, candidate.page_index - surrounding_pages)
        stop = min(page_count, candidate.page_index + surrounding_pages + 1)
        context_indexes.update(range(start, stop))

    return [_make_candidate(index, pages[index]) for index in sorted(context_indexes)]


def _format_items(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "—"


def main(
    pdf_path: Path = typer.Argument(..., help="NI 43-101 PDF 路径。"),
    top_k: int = typer.Option(5, "--top-k", min=1, help="返回候选页数量。"),
) -> None:
    """输出最可能包含 Mineral Resource 表格的候选页。"""

    candidates = locate_candidate_pages(pdf_path, top_k=top_k)
    table = Table(title=f"Mineral Resource 候选页：{pdf_path.name}")
    table.add_column("页码", justify="right")
    table.add_column("评分", justify="right")
    table.add_column("正向关键词")
    table.add_column("负向关键词")
    table.add_column("评分原因")

    for candidate in candidates:
        table.add_row(
            str(candidate.page_number),
            f"{candidate.score:.1f}",
            _format_items(candidate.matched_positive_keywords),
            _format_items(candidate.matched_negative_keywords),
            "; ".join(candidate.reasons),
        )

    Console().print(table)


if __name__ == "__main__":
    typer.run(main)
