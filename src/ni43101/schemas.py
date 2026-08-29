"""Pydantic schemas for NI 43-101 Mineral Resource extraction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ResourceCategory = Literal["Indicated", "Inferred"]
Commodity = Literal["Au", "Cu"]
GradeUnit = Literal["g/t Au", "% Cu"]
MetalUnit = Literal["oz", "t"]


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RawResourceRecord(_Schema):
    """A source-faithful record before any unit conversion."""

    document_id: str = Field(min_length=1)
    location: str = Field(min_length=1)
    category: ResourceCategory
    commodity: Commodity

    tonnage_value: float = Field(ge=0)
    tonnage_unit: str = Field(min_length=1)
    grade_value: float = Field(ge=0)
    grade_unit: str = Field(min_length=1)
    metal_value: float = Field(ge=0)
    metal_unit: str = Field(min_length=1)

    source_page: int = Field(ge=1)
    table_title: str | None = None
    evidence_text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    uncertain: bool = False
    uncertainty_reason: str | None = None


class ResourceRecord(_Schema):
    """A normalized Mineral Resource record ready for validation."""

    document_id: str = Field(min_length=1)
    location: str = Field(min_length=1)
    category: ResourceCategory
    commodity: Commodity

    tonnage_mt: float = Field(ge=0)
    grade: float = Field(ge=0)
    grade_unit: GradeUnit
    contained_metal: float = Field(ge=0)
    metal_unit: MetalUnit

    source_page: int = Field(ge=1)
    table_title: str | None = None
    evidence_text: str = Field(min_length=1)
    source_values: dict[str, Any]
    confidence: float = Field(ge=0, le=1)


class ExtractionResult(_Schema):
    """Raw records and document-level extraction diagnostics."""

    document_id: str = Field(min_length=1)
    candidate_pages: list[int]
    records: list[RawResourceRecord]
    uncertain: bool
    uncertainty_reasons: list[str]
    input_mode: str = Field(min_length=1)
