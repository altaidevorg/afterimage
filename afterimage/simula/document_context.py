"""Bounded document excerpts for taxonomy construction (paper input S / y)."""

from __future__ import annotations

from dataclasses import dataclass

from ..providers import DocumentProvider
from ..types import Document
from .types import DocumentProvenance, sha256_text


@dataclass
class BoundedDocContext:
    """Text blocks with provenance for prompts."""

    blocks: list[tuple[str, str]]  # (document_id, excerpt)
    provenance: list[DocumentProvenance]

    def prompt_block(self) -> str:
        parts: list[str] = []
        for doc_id, excerpt in self.blocks:
            parts.append(f"--- document_id={doc_id} ---\n{excerpt}")
        return "\n\n".join(parts)


def build_bounded_doc_context(
    provider: DocumentProvider | None,
    *,
    max_documents: int = 8,
    max_chars_per_doc: int = 4000,
    max_total_chars: int = 24000,
) -> BoundedDocContext:
    """Sample or load documents and truncate to explicit budgets."""
    if provider is None:
        return BoundedDocContext(blocks=[], provenance=[])

    docs: list[Document] = provider.get_all()
    if not docs:
        return BoundedDocContext(blocks=[], provenance=[])

    blocks: list[tuple[str, str]] = []
    provenance: list[DocumentProvenance] = []
    total = 0
    for doc in docs[:max_documents]:
        text = (doc.text or "").strip()
        if not text:
            continue
        excerpt = text[:max_chars_per_doc]
        if total + len(excerpt) > max_total_chars:
            room = max(0, max_total_chars - total)
            if room == 0:
                break
            excerpt = excerpt[:room]
        h = sha256_text(excerpt)
        blocks.append((doc.id, excerpt))
        provenance.append(
            DocumentProvenance(
                document_id=doc.id,
                excerpt_sha256=h,
                char_start=0,
                char_end=len(excerpt),
            )
        )
        total += len(excerpt)
        if total >= max_total_chars:
            break

    return BoundedDocContext(blocks=blocks, provenance=provenance)
