"""DeepSeek-backed extraction of raw NI 43-101 Mineral Resource records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson

from ni43101.config import Settings, get_settings
from ni43101.fewshot import FewShotProvider
from ni43101.llm_client import (
    LLMClientError,
    OpenAICompatibleClient,
    StructuredOutputError,
    StructuredCompletionClient,
)
from ni43101.pdf_locator import (
    CandidatePage,
    get_candidate_context,
    locate_candidate_pages,
)
from ni43101.schemas import ExtractionResult


EXTRACTOR_SYSTEM_PROMPT = """You are a strict NI 43-101 Mineral Resource table extraction agent.

Correctness is more important than completeness.

Only extract:
- Indicated Mineral Resources
- Inferred Mineral Resources

Never extract as target:
- Measured
- Measured + Indicated
- Proven
- Probable
- Mineral Reserves

Mineral Resources and Mineral Reserves are different concepts.

NI 43-101 tables may contain separate column groups for Measured, Indicated,
Measured + Indicated, and Inferred. Each group may contain Tonnage, Grade, and
Contained Metal. Do NOT confuse Indicated with Measured + Indicated. Preserve
column alignment exactly.

A table may contain multiple locations such as Tanami UG, Tanami OP, and Total
UG + OP, or Western Porphyries, Tanjeel, and Reko Diq Total. Do not return only
the Total row. Create a separate record for every explicit location, target
category, and commodity combination.

A table may contain separate Cu and Au sections. For Cu, use Tonnes, Cu %, and
Contained Cu from the same section. For Au, use Tonnes, Au g/t, and Contained Au
from the same section. Never combine fields across commodity sections.

Preserve source values and source units exactly. Do not normalize or calculate
units. For example, source values of 8000 kt, 3.40 g/t, and 870 koz must remain
tonnage_value=8000, tonnage_unit="kt", grade_value=3.40,
grade_unit="g/t Au", metal_value=870, and metal_unit="koz".

Every record must include source_page, table_title, and evidence_text.
evidence_text should contain the source table title, relevant headers, and row
text supporting that record. Never guess a number. If a record cannot be
confirmed, set uncertain=true and provide uncertainty_reason.

Return only an ExtractionResult JSON object. Do not output Markdown."""


EXTRACTION_TASK = """Extract source-faithful RawResourceRecord objects from the
provided candidate pages and their neighboring context. Extract every explicit
location/category/commodity target record, retain the original units, and cite
the supporting page and evidence. Exclude all non-target categories and all
Mineral Reserve records."""


EXTRACTOR_REVISION_PROMPT = """You are revising a previous extraction.

Re-check the ORIGINAL SOURCE EVIDENCE.

Do not blindly accept the Critic's suggested value.

The Critic only identifies risks.

Any corrected numeric value must be independently verified against the source table.

Only modify records necessary to fix supported issues.

If the evidence remains ambiguous:
set uncertain=true.

Do not guess."""


class ExtractorError(RuntimeError):
    """Controlled extraction failure suitable for handling by a CLI."""

    def __init__(self, document_id: str, message: str) -> None:
        super().__init__(f"extraction failed for {document_id}: {message}")
        self.document_id = document_id


class ExtractorConfigurationError(ExtractorError):
    """Raised when the DeepSeek extractor configuration is incomplete."""


class ExtractorOutputError(ExtractorError):
    """Raised when model output is malformed or outside the supplied evidence."""


class DeepSeekExtractor:
    """Build a bounded page prompt and request one raw ExtractionResult."""

    def __init__(
        self,
        client: StructuredCompletionClient,
        *,
        fewshot_provider: FewShotProvider | None = None,
    ) -> None:
        self.client = client
        self.fewshot_provider = fewshot_provider

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        fewshot_provider: FewShotProvider | None = None,
    ) -> DeepSeekExtractor:
        resolved = settings or get_settings()
        if resolved.extractor_api_key is None:
            raise ExtractorConfigurationError(
                "settings",
                "EXTRACTOR_API_KEY is not configured",
            )
        return cls(
            OpenAICompatibleClient(
                api_key=resolved.extractor_api_key.get_secret_value(),
                base_url=resolved.extractor_base_url,
                model=resolved.extractor_model,
            ),
            fewshot_provider=fewshot_provider,
        )

    def extract(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage] | None = None,
        *,
        input_mode: str = "candidate_pages_with_context",
    ) -> ExtractionResult:
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        selected_candidates = self._top_candidates(candidate_pages)
        if not selected_candidates:
            raise ExtractorError(document_id, "no candidate pages were provided")

        selected_context = self._neighbor_context(
            selected_candidates,
            context_pages or (),
        )
        messages = self._build_messages(
            document_id,
            selected_candidates,
            selected_context,
            input_mode,
            self._fewshot_prompt(document_id, selected_candidates),
        )

        try:
            result = self.client.structured_completion(messages, ExtractionResult)
        except StructuredOutputError as error:
            raise ExtractorOutputError(document_id, str(error)) from error
        except LLMClientError as error:
            raise ExtractorError(document_id, str(error)) from error

        self._validate_result_scope(
            document_id,
            result,
            selected_candidates,
            selected_context,
        )
        return result

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        *,
        document_id: str | None = None,
    ) -> ExtractionResult:
        """Locate Top-3 pages, add one-page context, then run extraction."""

        path = Path(pdf_path)
        resolved_document_id = document_id or path.stem
        candidates = locate_candidate_pages(path, top_k=3)
        context = get_candidate_context(path, candidates, surrounding_pages=1)
        return self.extract(
            resolved_document_id,
            candidates,
            context,
            input_mode="candidate_pages_with_context",
        )

    def revise(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage] | None,
        previous_extraction: ExtractionResult,
        validator_failures: Sequence[Any],
        critic_issues: Sequence[Any],
        *,
        input_mode: str = "candidate_pages_with_context_revision",
    ) -> ExtractionResult:
        """Revise one extraction using source evidence and audit feedback."""

        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if previous_extraction.document_id != document_id:
            raise ExtractorError(
                document_id,
                "previous extraction document_id does not match input",
            )

        selected_candidates = self._top_candidates(candidate_pages)
        if not selected_candidates:
            raise ExtractorError(document_id, "no candidate pages were provided")
        selected_context = self._neighbor_context(
            selected_candidates,
            context_pages or (),
        )
        messages = self._build_revision_messages(
            document_id=document_id,
            candidates=selected_candidates,
            context_pages=selected_context,
            previous_extraction=previous_extraction,
            validator_failures=validator_failures,
            critic_issues=critic_issues,
            input_mode=input_mode,
            fewshot_prompt=self._fewshot_prompt(document_id, selected_candidates),
        )

        try:
            result = self.client.structured_completion(messages, ExtractionResult)
        except StructuredOutputError as error:
            raise ExtractorOutputError(document_id, str(error)) from error
        except LLMClientError as error:
            raise ExtractorError(document_id, str(error)) from error

        self._validate_result_scope(
            document_id,
            result,
            selected_candidates,
            selected_context,
        )
        return result

    @staticmethod
    def _top_candidates(
        candidate_pages: Sequence[CandidatePage],
    ) -> list[CandidatePage]:
        ranked = sorted(
            candidate_pages,
            key=lambda page: (-page.score, page.page_index),
        )
        unique: list[CandidatePage] = []
        seen_indexes: set[int] = set()
        for page in ranked:
            if page.page_index in seen_indexes:
                continue
            unique.append(page)
            seen_indexes.add(page.page_index)
            if len(unique) == 3:
                break
        return unique

    @staticmethod
    def _neighbor_context(
        candidates: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage],
    ) -> list[CandidatePage]:
        candidate_indexes = {page.page_index for page in candidates}
        allowed_context_indexes = {
            neighbor_index
            for page_index in candidate_indexes
            for neighbor_index in (page_index - 1, page_index + 1)
            if neighbor_index >= 0
        }

        by_index: dict[int, CandidatePage] = {}
        for page in context_pages:
            if (
                page.page_index in allowed_context_indexes
                and page.page_index not in candidate_indexes
            ):
                by_index.setdefault(page.page_index, page)
        return [by_index[index] for index in sorted(by_index)]

    @staticmethod
    def _build_messages(
        document_id: str,
        candidates: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage],
        input_mode: str,
        fewshot_prompt: str = "",
    ) -> list[dict[str, str]]:
        payload: dict[str, Any] = {
            "document_id": document_id,
            "task": EXTRACTION_TASK,
            "input_mode": input_mode,
            "candidate_pages": [
                {"page_number": page.page_number, "text": page.text}
                for page in candidates
            ],
            "context_pages": [
                {"page_number": page.page_number, "text": page.text}
                for page in context_pages
            ],
            "output_schema": ExtractionResult.model_json_schema(),
        }
        if fewshot_prompt:
            payload["evolution_fewshot"] = fewshot_prompt
        return [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode("utf-8"),
            },
        ]

    @classmethod
    def _build_revision_messages(
        cls,
        *,
        document_id: str,
        candidates: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage],
        previous_extraction: ExtractionResult,
        validator_failures: Sequence[Any],
        critic_issues: Sequence[Any],
        input_mode: str,
        fewshot_prompt: str = "",
    ) -> list[dict[str, str]]:
        payload: dict[str, Any] = {
            "document_id": document_id,
            "task": EXTRACTOR_REVISION_PROMPT,
            "input_mode": input_mode,
            "original_source_evidence": {
                "candidate_pages": [
                    {"page_number": page.page_number, "text": page.text}
                    for page in candidates
                ],
                "context_pages": [
                    {"page_number": page.page_number, "text": page.text}
                    for page in context_pages
                ],
            },
            "previous_extraction": previous_extraction.model_dump(mode="json"),
            "validator_failures": [
                cls._json_value(value) for value in validator_failures
            ],
            "critic_issues": [cls._json_value(value) for value in critic_issues],
            "output_schema": ExtractionResult.model_json_schema(),
        }
        if fewshot_prompt:
            payload["evolution_fewshot"] = fewshot_prompt
        return [
            {
                "role": "system",
                "content": f"{EXTRACTOR_SYSTEM_PROMPT}\n\n{EXTRACTOR_REVISION_PROMPT}",
            },
            {
                "role": "user",
                "content": orjson.dumps(payload).decode("utf-8"),
            },
        ]

    @staticmethod
    def _json_value(value: Any) -> Any:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        return value

    def _fewshot_prompt(
        self,
        document_id: str,
        candidate_pages: Sequence[CandidatePage],
    ) -> str:
        if self.fewshot_provider is None:
            return ""
        return self.fewshot_provider.render(document_id, candidate_pages)

    @staticmethod
    def _validate_result_scope(
        document_id: str,
        result: ExtractionResult,
        candidates: Sequence[CandidatePage],
        context_pages: Sequence[CandidatePage],
    ) -> None:
        if result.document_id != document_id:
            raise ExtractorOutputError(
                document_id,
                "response document_id does not match input",
            )

        available_pages = {
            page.page_number for page in [*candidates, *context_pages]
        }
        for record in result.records:
            if record.document_id != document_id:
                raise ExtractorOutputError(
                    document_id,
                    "record document_id does not match input",
                )
            if record.source_page not in available_pages:
                raise ExtractorOutputError(
                    document_id,
                    f"record cites unsupplied source page {record.source_page}",
                )
            if record.uncertain and not record.uncertainty_reason:
                raise ExtractorOutputError(
                    document_id,
                    "uncertain record is missing uncertainty_reason",
                )
