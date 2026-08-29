"""Leakage-safe replay of at most three evolution lessons."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ni43101.evolution import (
    DEFAULT_EVOLUTION_PATH,
    LESSONS,
    EvolutionLogger,
    EvolutionRecord,
    FailureType,
)
from ni43101.pdf_locator import CandidatePage


MAX_FEWSHOT_CASES = 3


class FewShotProvider(Protocol):
    def render(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
    ) -> str: ...


class FewShotQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    commodities: set[str] = Field(default_factory=set)
    failure_types: set[FailureType] = Field(default_factory=set)
    source_units: set[str] = Field(default_factory=set)
    table_structure_risks: set[str] = Field(default_factory=set)


class SelectedEvolutionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    document_id: str
    similarity_score: float
    failure_types: list[FailureType]
    prompt: str


_MISTAKES: dict[FailureType, str] = {
    FailureType.RESOURCE_RESERVE_CONFUSION: (
        "A Mineral Reserve or Proven/Probable table was treated as a target "
        "Mineral Resource table."
    ),
    FailureType.CATEGORY_COLUMN_SHIFT: (
        "A numeric triplet was assigned to the wrong resource category header."
    ),
    FailureType.MEASURED_PLUS_INDICATED_CONFUSION: (
        "Measured + Indicated was mistaken for the standalone Indicated category."
    ),
    FailureType.ROW_ALIGNMENT_ERROR: (
        "Location, tonnage, grade, and contained metal were taken from different rows."
    ),
    FailureType.TONNAGE_UNIT_ERROR: (
        "The source ore tonnage unit was lost or converted during extraction."
    ),
    FailureType.METAL_UNIT_ERROR: (
        "A contained-metal unit was interpreted as an ore-tonnage unit."
    ),
    FailureType.COMMODITY_MISMATCH: (
        "Fields from separate Au and Cu sections were combined."
    ),
    FailureType.MATH_INCONSISTENCY: (
        "The extracted tonnage-grade-metal triplet failed deterministic arithmetic."
    ),
    FailureType.MISSING_EVIDENCE: (
        "A value was accepted without direct support in the supplied source page."
    ),
    FailureType.AMBIGUOUS_TABLE: (
        "Ambiguous headers or rows were treated as certain."
    ),
    FailureType.LLM_FORMAT_ERROR: (
        "The extraction response did not satisfy the required JSON schema."
    ),
    FailureType.MISSING_CATEGORY: (
        "A clearly present Indicated or Inferred target category was omitted."
    ),
    FailureType.LOW_CRITIC_SCORE: (
        "An extraction retained unresolved risks and did not reach the safety score."
    ),
}


_CRITIC_FINDINGS: dict[FailureType, str] = {
    FailureType.RESOURCE_RESERVE_CONFUSION: (
        "Resource and Reserve semantics were not kept separate."
    ),
    FailureType.CATEGORY_COLUMN_SHIFT: "The values did not align with their parent header.",
    FailureType.MEASURED_PLUS_INDICATED_CONFUSION: (
        "Standalone Indicated and aggregate Measured + Indicated are distinct columns."
    ),
    FailureType.ROW_ALIGNMENT_ERROR: "The record fields were not supported by one row.",
    FailureType.TONNAGE_UNIT_ERROR: "The raw ore unit was not preserved.",
    FailureType.METAL_UNIT_ERROR: "Contained-metal normalization used the wrong semantics.",
    FailureType.COMMODITY_MISMATCH: "Commodity, grade unit, and metal unit were incompatible.",
    FailureType.MATH_INCONSISTENCY: "The deterministic relative error exceeded tolerance.",
    FailureType.MISSING_EVIDENCE: "The cited evidence did not support every core value.",
    FailureType.AMBIGUOUS_TABLE: "The evidence required revision or abstention.",
    FailureType.LLM_FORMAT_ERROR: "The response was not schema-valid JSON.",
    FailureType.MISSING_CATEGORY: "The target table was not extracted completely.",
    FailureType.LOW_CRITIC_SCORE: "The audit found unresolved extraction risk.",
}


_FAILURE_RISKS: dict[FailureType, set[str]] = {
    FailureType.RESOURCE_RESERVE_CONFUSION: {"resource_vs_reserve"},
    FailureType.CATEGORY_COLUMN_SHIFT: {"category_columns"},
    FailureType.MEASURED_PLUS_INDICATED_CONFUSION: {"measured_plus_indicated"},
    FailureType.ROW_ALIGNMENT_ERROR: {"row_alignment"},
    FailureType.TONNAGE_UNIT_ERROR: {"unit_alignment"},
    FailureType.METAL_UNIT_ERROR: {"unit_alignment"},
    FailureType.COMMODITY_MISMATCH: {"multi_commodity"},
    FailureType.MATH_INCONSISTENCY: {"numeric_alignment"},
    FailureType.MISSING_EVIDENCE: {"evidence_support"},
    FailureType.AMBIGUOUS_TABLE: {"ambiguous_table"},
    FailureType.MISSING_CATEGORY: {"category_columns"},
}


class EvolutionFewShotSelector:
    """Rank a frozen evolution snapshot and emit pattern-only examples."""

    def __init__(
        self,
        path: str | Path = DEFAULT_EVOLUTION_PATH,
        *,
        max_cases: int = MAX_FEWSHOT_CASES,
        exclude_same_document: bool = True,
        records: Sequence[EvolutionRecord] | None = None,
    ) -> None:
        if not 1 <= max_cases <= MAX_FEWSHOT_CASES:
            raise ValueError(f"max_cases must be between 1 and {MAX_FEWSHOT_CASES}")
        self.max_cases = max_cases
        self.exclude_same_document = exclude_same_document
        # Freeze the snapshot so an evaluation run cannot learn from cases it
        # generates later in that same run.
        self.records = list(records) if records is not None else EvolutionLogger(path).read()

    def select(self, query: FewShotQuery) -> list[SelectedEvolutionCase]:
        ranked: list[tuple[float, float, EvolutionRecord]] = []
        query_document = _normalized_document_id(query.document_id)
        for record in self.records:
            if (
                self.exclude_same_document
                and _normalized_document_id(record.document_id) == query_document
            ):
                continue
            score = self._similarity(query, record)
            if score <= 0:
                continue
            ranked.append((score, record.timestamp.timestamp(), record))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].case_id))
        return [
            SelectedEvolutionCase(
                case_id=record.case_id,
                document_id=record.document_id,
                similarity_score=score,
                failure_types=list(record.failure_types),
                prompt=_safe_case_prompt(record.failure_types),
            )
            for score, _, record in ranked[: self.max_cases]
        ]

    def render(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
    ) -> str:
        query = query_from_candidate_pages(document_id, candidate_pages)
        selected = self.select(query)
        if not selected:
            return ""
        header = (
            "Historical error-pattern reminders only. These examples contain no "
            "Ground Truth answers. Re-check the current source independently."
        )
        return "\n\n".join([header, *(case.prompt for case in selected)])

    @staticmethod
    def _similarity(query: FewShotQuery, record: EvolutionRecord) -> float:
        record_commodities = _as_set(record.commodity)
        record_failures = set(record.failure_types)
        record_units = _record_source_units(record)
        record_risks = {
            risk
            for failure_type in record_failures
            for risk in _FAILURE_RISKS.get(failure_type, set())
        }
        return (
            5.0 * len(query.commodities & record_commodities)
            + 4.0 * len(query.failure_types & record_failures)
            + 2.0 * len(query.source_units & record_units)
            + 3.0 * len(query.table_structure_risks & record_risks)
        )


def query_from_candidate_pages(
    document_id: str,
    candidate_pages: Sequence[CandidatePage],
) -> FewShotQuery:
    text = "\n".join(page.text for page in candidate_pages).casefold()
    commodities: set[str] = set()
    if any(token in text for token in ("g/t", " au", "gold")):
        commodities.add("Au")
    if any(token in text for token in ("% cu", "cu %", "copper", "contained cu")):
        commodities.add("Cu")

    units = {
        match.casefold()
        for match in re.findall(r"(?<![a-z])(?:t|kt|mt|oz|koz|moz|g/t)(?![a-z])", text)
    }
    failures: set[FailureType] = set()
    risks: set[str] = set()
    if "measured + indicated" in text or "measured and indicated" in text:
        failures.add(FailureType.MEASURED_PLUS_INDICATED_CONFUSION)
        risks.add("measured_plus_indicated")
    if any(token in text for token in ("mineral reserve", "proven", "probable")):
        failures.add(FailureType.RESOURCE_RESERVE_CONFUSION)
        risks.add("resource_vs_reserve")
    if len(commodities) > 1:
        failures.add(FailureType.COMMODITY_MISMATCH)
        risks.add("multi_commodity")
    if "indicated" in text and "inferred" in text:
        risks.add("category_columns")
    if "contained" in text and any(unit in units for unit in ("kt", "mt", "koz", "moz")):
        risks.add("unit_alignment")

    return FewShotQuery(
        document_id=document_id,
        commodities=commodities,
        failure_types=failures,
        source_units=units,
        table_structure_risks=risks,
    )


def _safe_case_prompt(failure_types: Sequence[FailureType]) -> str:
    blocks: list[str] = []
    for failure_type in failure_types:
        if failure_type not in _MISTAKES:
            continue
        blocks.append(
            "\n".join(
                [
                    f"Previous extraction mistake: {_MISTAKES[failure_type]}",
                    f"Critic finding: {_CRITIC_FINDINGS[failure_type]}",
                    f"Lesson: {LESSONS[failure_type]}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _record_source_units(record: EvolutionRecord) -> set[str]:
    output = record.extractor_output or {}
    raw_records = output.get("records", [])
    if not isinstance(raw_records, list):
        return set()
    units: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        for field in ("tonnage_unit", "grade_unit", "metal_unit"):
            value = raw_record.get(field)
            if isinstance(value, str):
                normalized = value.strip().casefold()
                units.add(normalized)
                units.update(
                    re.findall(
                        r"(?<![a-z])(?:t|kt|mt|oz|koz|moz|g/t)(?![a-z])",
                        normalized,
                    )
                )
    return units


def _as_set(value: str | list[str] | None) -> set[str]:
    if value is None:
        return set()
    return {value} if isinstance(value, str) else set(value)


def _normalized_document_id(document_id: str) -> str:
    return " ".join(document_id.split()).casefold()
