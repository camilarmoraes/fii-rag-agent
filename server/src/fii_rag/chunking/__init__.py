"""Pacote de estratégias de chunking.

Re-exporta também os símbolos legados (`LangChainParser`,
`LangChainSemanticExtractor`, `ExtractedMetadata`) de `legacy.py` para que os
imports antigos (`from server.src.fii_rag.chunking import LangChainParser`)
continuem funcionando até o cleanup do PR 6.
"""

from server.src.fii_rag.chunking.base import (
    Chunk,
    DocContext,
    IChunkingStrategy,
    LoadedDocument,
    PageText,
)
from server.src.fii_rag.chunking.doc_aware import DocAwareChunker
from server.src.fii_rag.chunking.hierarchical import HierarchicalChunker
from server.src.fii_rag.chunking.legacy import (
    ExtractedMetadata,
    LangChainParser,
    LangChainSemanticExtractor,
)
from server.src.fii_rag.chunking.recursive import RecursiveChunker
from server.src.fii_rag.chunking.registry import ALL_STRATEGIES, build_strategy
from server.src.fii_rag.chunking.semantic import SemanticChunker
from server.src.fii_rag.chunking.summary import DocumentSummaryChunker

__all__ = [
    "ALL_STRATEGIES",
    "Chunk",
    "DocAwareChunker",
    "DocContext",
    "DocumentSummaryChunker",
    # Legacy re-exports
    "ExtractedMetadata",
    "HierarchicalChunker",
    "IChunkingStrategy",
    "LangChainParser",
    "LangChainSemanticExtractor",
    "LoadedDocument",
    "PageText",
    "RecursiveChunker",
    "SemanticChunker",
    "build_strategy",
]
