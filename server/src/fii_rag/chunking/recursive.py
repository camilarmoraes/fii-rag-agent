"""Recursive character chunker — baseline rápido, preserva `page_number`."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from server.src.fii_rag.chunking.base import (
    Chunk,
    DocContext,
    IChunkingStrategy,
    LoadedDocument,
)


class RecursiveChunker(IChunkingStrategy):
    name = "recursive"
    produces_one_per_doc = False

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, doc: LoadedDocument, ctx: DocContext) -> list[Chunk]:
        """Split por página para preservar `page_number` no payload."""
        chunks: list[Chunk] = []
        for page in doc.pages:
            for text in self.splitter.split_text(page.text):
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        text=text,
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                    )
                )
        return chunks
