"""Tipos compartilhados pelas estratégias de chunking.

`IChunkingStrategy` é a interface implementada por todas as estratégias do
pacote `chunking/`. Cada implementação recebe um `LoadedDocument` (saída do
`PdfLoader`) + `DocContext` (metadados-doc para herdar nos chunks) e devolve
uma lista de `Chunk`s parciais — o `ChunkEnricher` completa depois com
metadados LLM, numerics e identificadores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from server.src.fii_rag.schemas.chunk import ChunkingStrategyName
from server.src.fii_rag.schemas.document import DocumentMetadata


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass
class LoadedDocument:
    full_text: str
    pages: list[PageText]
    markdown: str  # de pymupdf4llm; pode ser "" se a extração falhou
    total_pages: int
    source_path: str


@dataclass
class DocContext:
    doc_id: str
    doc_metadata: DocumentMetadata


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None


class IChunkingStrategy(ABC):
    """Interface comum a todas as estratégias de chunking."""

    name: ChunkingStrategyName  # subclasses devem definir
    produces_one_per_doc: bool = False  # True para summary

    @abstractmethod
    def chunk(self, doc: LoadedDocument, ctx: DocContext) -> list[Chunk]:
        ...
