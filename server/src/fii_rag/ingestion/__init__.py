"""Pacote de ingestão.

Re-exporta `PDFIngestionManager` (legado) para preservar
`from server.src.fii_rag.ingestion import PDFIngestionManager`.
"""

from server.src.fii_rag.ingestion.enricher import ChunkEnricher
from server.src.fii_rag.ingestion.legacy import PDFIngestionManager
from server.src.fii_rag.ingestion.loader import PdfLoader
from server.src.fii_rag.ingestion.pipeline import (
    IngestionPipeline,
    build_ingestion_pipeline,
    compute_doc_id,
)

__all__ = [
    "ChunkEnricher",
    "IngestionPipeline",
    "PDFIngestionManager",
    "PdfLoader",
    "build_ingestion_pipeline",
    "compute_doc_id",
]
