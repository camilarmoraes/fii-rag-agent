"""Document-aware chunker — quebra preservando estrutura de seções/headings.

Usa o markdown gerado pelo `pymupdf4llm` (em `LoadedDocument.markdown`) para
identificar headings (linhas `#`, `##`, `###`...) e agrupar conteúdo por seção.
Seções > `max_chunk_size` caem em recursive splitter mantendo o heading no
metadata. Se markdown estiver vazio (PDF sem estrutura), cai num split
recursivo simples sobre `full_text`.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from server.src.fii_rag.chunking.base import (
    Chunk,
    DocContext,
    IChunkingStrategy,
    LoadedDocument,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class DocAwareChunker(IChunkingStrategy):
    name = "doc_aware"
    produces_one_per_doc = False

    def __init__(self, max_chunk_size: int = 2000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, doc: LoadedDocument, ctx: DocContext) -> list[Chunk]:
        if not doc.markdown.strip():
            # Fallback: PDF sem estrutura de heading
            return self._chunk_plaintext(doc.full_text)

        sections = self._split_by_heading(doc.markdown)
        if not sections:
            return self._chunk_plaintext(doc.full_text)

        chunks: list[Chunk] = []
        for heading, text in sections:
            text = text.strip()
            if not text:
                continue
            if len(text) <= self.max_chunk_size:
                chunks.append(
                    Chunk(
                        text=text,
                        chunk_index=len(chunks),
                        section_heading=heading or None,
                    )
                )
            else:
                for sub in self.fallback_splitter.split_text(text):
                    if not sub.strip():
                        continue
                    chunks.append(
                        Chunk(
                            text=sub,
                            chunk_index=len(chunks),
                            section_heading=heading or None,
                        )
                    )
        return chunks

    def _chunk_plaintext(self, text: str) -> list[Chunk]:
        return [
            Chunk(text=t, chunk_index=i)
            for i, t in enumerate(self.fallback_splitter.split_text(text))
            if t.strip()
        ]

    @staticmethod
    def _split_by_heading(md: str) -> list[tuple[str, str]]:
        """Devolve `[(heading, body)]` na ordem do documento."""
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_body: list[str] = []
        for line in md.splitlines():
            m = HEADING_RE.match(line)
            if m:
                if current_body:
                    sections.append((current_heading, "\n".join(current_body)))
                    current_body = []
                current_heading = m.group(2).strip()
            else:
                current_body.append(line)
        if current_body:
            sections.append((current_heading, "\n".join(current_body)))
        return [s for s in sections if s[1].strip()]
