"""Registry das estratégias disponíveis + fábrica que usa o `AppConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from server.src.fii_rag.chunking.base import IChunkingStrategy
from server.src.fii_rag.chunking.doc_aware import DocAwareChunker
from server.src.fii_rag.chunking.hierarchical import HierarchicalChunker
from server.src.fii_rag.chunking.recursive import RecursiveChunker
from server.src.fii_rag.chunking.semantic import SemanticChunker
from server.src.fii_rag.chunking.summary import DocumentSummaryChunker

if TYPE_CHECKING:
    from server.src.fii_rag.config import AppConfig

ALL_STRATEGIES: list[str] = [
    "summary",
    "recursive",
    "semantic",
    "doc_aware",
    "hierarchical",
]


def build_strategy(
    name: str, config: "AppConfig", embeddings: Any = None
) -> IChunkingStrategy:
    """Constrói uma `IChunkingStrategy` pelo nome, lendo parâmetros do `AppConfig`.

    `embeddings` é obrigatório para a estratégia `semantic` — passe o objeto
    `Embeddings` LangChain já configurado (mesmo usado no upsert).
    """
    if name == "summary":
        return DocumentSummaryChunker()
    if name == "recursive":
        return RecursiveChunker(
            chunk_size=config.chunking.recursive_chunk_size,
            chunk_overlap=config.chunking.recursive_chunk_overlap,
        )
    if name == "semantic":
        if embeddings is None:
            raise ValueError(
                "SemanticChunker requer um embedder LangChain (passe `embeddings=...`)."
            )
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=config.chunking.semantic_breakpoint_threshold_type,
            breakpoint_threshold_amount=config.chunking.semantic_breakpoint_threshold_amount,
        )
    if name == "doc_aware":
        return DocAwareChunker(
            max_chunk_size=config.chunking.recursive_chunk_size * 2,
            chunk_overlap=config.chunking.recursive_chunk_overlap,
        )
    if name == "hierarchical":
        return HierarchicalChunker(
            parent_size=config.chunking.hierarchical_parent_size,
            child_size=config.chunking.hierarchical_child_size,
            child_overlap=config.chunking.hierarchical_child_overlap,
        )
    raise ValueError(
        f"Estratégia de chunking desconhecida: {name!r}. "
        f"Disponíveis: {ALL_STRATEGIES}."
    )
