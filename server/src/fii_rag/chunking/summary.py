"""Document-summary chunker — 1 chunk por documento, texto = sumário gerado pelo LLM."""

from __future__ import annotations

from server.src.fii_rag.chunking.base import (
    Chunk,
    DocContext,
    IChunkingStrategy,
    LoadedDocument,
)


class DocumentSummaryChunker(IChunkingStrategy):
    name = "summary"
    produces_one_per_doc = True

    def chunk(self, doc: LoadedDocument, ctx: DocContext) -> list[Chunk]:
        summary = ctx.doc_metadata.summary or ""
        if not summary.strip():
            # Fallback: usa o início do doc para não inserir ponto vazio
            summary = doc.full_text[:2000]
        return [Chunk(text=summary, chunk_index=0)]
