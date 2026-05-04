"""Semantic chunker via `langchain_experimental` — quebra por mudança semântica."""

from __future__ import annotations

from typing import Any

from server.src.fii_rag.chunking.base import (
    Chunk,
    DocContext,
    IChunkingStrategy,
    LoadedDocument,
)


class SemanticChunker(IChunkingStrategy):
    name = "semantic"
    produces_one_per_doc = False

    def __init__(
        self,
        embeddings: Any,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
    ):
        from langchain_experimental.text_splitter import (
            SemanticChunker as LCSemanticChunker,
        )

        self.splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def chunk(self, doc: LoadedDocument, ctx: DocContext) -> list[Chunk]:
        """Roda no `full_text` — `page_number` não é preservado nesta strategy."""
        if not doc.full_text.strip():
            return []
        texts = self.splitter.split_text(doc.full_text)
        return [
            Chunk(text=t, chunk_index=i)
            for i, t in enumerate(texts)
            if t.strip()
        ]
