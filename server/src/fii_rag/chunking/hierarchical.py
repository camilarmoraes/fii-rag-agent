"""Parent-child chunker — indexa filhos pequenos, mas carrega o pai inline.

Estratégia clássica de RAG longo: chunks pequenos (filhos) ganham precisão
na busca; mas o LLM final recebe o pai (~2000 chars) para ter contexto largo.
Aqui guardamos `parent_text` (truncado em 1500 chars) direto no payload do
filho — `TwoStageRetriever` no PR 5 monta o contexto final concatenando.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from server.src.fii_rag.chunking.base import (
    Chunk,
    DocContext,
    IChunkingStrategy,
    LoadedDocument,
)

PARENT_TEXT_MAX_CHARS = 1500


class HierarchicalChunker(IChunkingStrategy):
    name = "hierarchical"
    produces_one_per_doc = False

    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 400,
        child_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=0
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=child_overlap
        )

    def chunk(self, doc: LoadedDocument, ctx: DocContext) -> list[Chunk]:
        if not doc.full_text.strip():
            return []
        chunks: list[Chunk] = []
        for parent_text in self.parent_splitter.split_text(doc.full_text):
            parent_text = parent_text.strip()
            if not parent_text:
                continue
            parent_id = str(uuid4())
            parent_truncated = parent_text[:PARENT_TEXT_MAX_CHARS]
            for child_text in self.child_splitter.split_text(parent_text):
                child_text = child_text.strip()
                if not child_text:
                    continue
                chunks.append(
                    Chunk(
                        text=child_text,
                        chunk_index=len(chunks),
                        parent_chunk_id=parent_id,
                        parent_text=parent_truncated,
                    )
                )
        return chunks
