"""Transforma `Chunk` parciais em `ChunkMetadata` completos.

Concatena 4 fontes de informação para cada chunk:
1. **Doc-level inherited** — campos do `DocumentMetadata` replicados em todo
   chunk filho (`doc_id`, `ticker`, `cnpj`, `report_*`, etc.).
2. **Chunk-level estrutural** — vindo do `Chunk` (chunk_index, page_number,
   section_heading, parent_*).
3. **Numerics** — extraídos por `NumericFactsExtractor` (regex + LLM fallback).
4. **LLM-extracted local** — `title`/`keywords`/`chunk_summary` via
   `ChunkMetadataExtractor.extract_batch` em paralelo.
"""

from __future__ import annotations

from uuid import uuid4

from server.src.fii_rag.chunking.base import Chunk, DocContext
from server.src.fii_rag.extraction.chunk_metadata import ChunkMetadataExtractor
from server.src.fii_rag.extraction.numerics import NumericFactsExtractor
from server.src.fii_rag.schemas.chunk import (
    ChunkExtractedMetadata,
    ChunkMetadata,
    ChunkingStrategyName,
)

EMPTY_LLM_META = ChunkExtractedMetadata(title="", keywords=[], chunk_summary="")


class ChunkEnricher:
    def __init__(
        self,
        chunk_metadata_extractor: ChunkMetadataExtractor,
        numerics_extractor: NumericFactsExtractor,
    ):
        self.chunk_metadata_extractor = chunk_metadata_extractor
        self.numerics_extractor = numerics_extractor

    def enrich(
        self,
        chunks: list[Chunk],
        ctx: DocContext,
        strategy_name: ChunkingStrategyName,
        run_chunk_metadata: bool = True,
    ) -> list[ChunkMetadata]:
        if not chunks:
            return []

        if run_chunk_metadata:
            llm_meta = self.chunk_metadata_extractor.extract_batch(
                [c.text for c in chunks]
            )
        else:
            llm_meta = [EMPTY_LLM_META] * len(chunks)

        doc = ctx.doc_metadata
        out: list[ChunkMetadata] = []
        for chunk, extracted in zip(chunks, llm_meta):
            numerics = self.numerics_extractor.extract(chunk.text)
            out.append(
                ChunkMetadata(
                    # doc-inherited
                    doc_id=doc.doc_id,
                    source_filename=doc.source_filename,
                    ticker=doc.ticker,
                    cnpj=doc.cnpj,
                    fund_name=doc.fund_name,
                    report_date=doc.report_date,
                    report_month=doc.report_month,
                    report_year=doc.report_year,
                    report_quarter=doc.report_quarter,
                    report_type=doc.report_type,
                    # chunk-level
                    chunk_id=str(uuid4()),
                    strategy=strategy_name,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    parent_chunk_id=chunk.parent_chunk_id,
                    parent_text=chunk.parent_text,
                    # llm-extracted
                    title=extracted.title,
                    keywords=list(extracted.keywords),
                    chunk_summary=extracted.chunk_summary,
                    # numerics
                    numerics=numerics,
                )
            )
        return out
