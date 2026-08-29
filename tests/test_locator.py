from pathlib import Path

import pytest

from ni43101 import pdf_locator


RESOURCE_TABLE_TEXT = """
Mineral Resource Statement
Classification Tonnes Grade Contained Au
Measured 1,000 2.0 64
Indicated 2,000 1.8 116
Measured + Indicated 3,000 1.9 180
Inferred 900 1.4 41
"""

RESERVE_TABLE_TEXT = """
Mineral Reserve Statement
Mineral Reserves are based on modifying factors.
Category Tonnage Grade Contained Au
Proven 1,000 2.0 64
Probable 2,000 1.8 116
Indicated Mineral Resource figures are shown only as a reference.
"""


def test_resource_table_scores_above_reserve_table() -> None:
    assert pdf_locator.score_page(RESOURCE_TABLE_TEXT) > 80
    assert pdf_locator.score_page(RESOURCE_TABLE_TEXT) > pdf_locator.score_page(
        RESERVE_TABLE_TEXT
    )


def test_indicated_and_inferred_combination_receives_large_bonus() -> None:
    indicated_only = "Mineral Resource Indicated Tonnes Grade Contained Au"
    both_categories = indicated_only + " Inferred"

    assert pdf_locator.score_page(both_categories) > (
        pdf_locator.score_page(indicated_only) + 30
    )


def test_locate_candidate_pages_ranks_pages_and_preserves_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        "General project introduction",
        RESERVE_TABLE_TEXT,
        RESOURCE_TABLE_TEXT,
        "Mineral Resources Summary Indicated and Inferred Tonnes Grade",
    ]
    monkeypatch.setattr(pdf_locator, "extract_pages", lambda _: pages)

    candidates = pdf_locator.locate_candidate_pages("mock.pdf", top_k=2)

    assert [candidate.page_number for candidate in candidates] == [3, 4]
    assert candidates[0].page_index == 2
    assert candidates[0].text == RESOURCE_TABLE_TEXT
    assert "Indicated" in candidates[0].matched_positive_keywords
    assert "Inferred" in candidates[0].matched_positive_keywords
    assert candidates[0].matched_negative_keywords == []


def test_candidate_context_includes_neighbors_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [f"page {index}" for index in range(6)]
    monkeypatch.setattr(pdf_locator, "extract_pages", lambda _: pages)
    candidates = [
        pdf_locator.CandidatePage(
            page_index=2,
            page_number=3,
            score=10,
            matched_positive_keywords=[],
            matched_negative_keywords=[],
            reasons=[],
            text=pages[2],
        ),
        pdf_locator.CandidatePage(
            page_index=3,
            page_number=4,
            score=9,
            matched_positive_keywords=[],
            matched_negative_keywords=[],
            reasons=[],
            text=pages[3],
        ),
    ]

    context = pdf_locator.get_candidate_context(Path("mock.pdf"), candidates)

    assert [page.page_index for page in context] == [1, 2, 3, 4]
    assert [page.text for page in context] == pages[1:5]


def test_top_k_and_context_radius_must_be_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_locator, "extract_pages", lambda _: ["page"])

    with pytest.raises(ValueError, match="top_k"):
        pdf_locator.locate_candidate_pages("mock.pdf", top_k=0)
    with pytest.raises(ValueError, match="surrounding_pages"):
        pdf_locator.get_candidate_context("mock.pdf", [], surrounding_pages=-1)
