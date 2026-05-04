"""Extração de metadados por chunk em batches paralelos via `CHUNK_METADATA_LLM_MODEL`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from server.src.fii_rag.schemas.chunk import ChunkExtractedMetadata

PROMPT = (
    "Analise o trecho de relatório de Fundo Imobiliário (FII) abaixo e extraia "
    "metadados resumidos do trecho:\n"
    "- title: título conciso (5-10 palavras)\n"
    "- keywords: 5+ palavras-chave para busca\n"
    "- chunk_summary: resumo curto (1-2 frases)\n\n"
    "--- TRECHO ---\n"
)

EMPTY = ChunkExtractedMetadata(title="", keywords=[], chunk_summary="")


class ChunkMetadataExtractor:
    def __init__(self, llm: Any, max_workers: int = 8):
        self.llm = llm.with_structured_output(ChunkExtractedMetadata)
        self.max_workers = max_workers

    def extract_batch(self, texts: list[str]) -> list[ChunkExtractedMetadata]:
        if not texts:
            return []

        def _one(t: str) -> ChunkExtractedMetadata:
            try:
                return self.llm.invoke(PROMPT + t)
            except Exception:
                return EMPTY

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(_one, texts))
