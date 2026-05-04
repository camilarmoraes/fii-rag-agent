"""Pipeline de ingestão multi-strategy para coleções lógicas.

Fluxo (`run(pdf_path, logical)`):
    1. `PdfLoader.load(pdf_path)` → `LoadedDocument` (texto + páginas + markdown)
    2. `provisioner.read_logical_config(logical)` — erro se ausente
    3. `DocMetadataExtractor.extract(full_text)` → metadata-doc + summary
       (1 chamada LLM); `doc_id` calculado deterministicamente
    4. Para cada strategy ativa, **em paralelo** (ThreadPoolExecutor):
       - constrói a strategy via `build_strategy(name, config, embeddings)`
       - chunk → enrich (metadata herdada + LLM batch + numerics) → embed
         dense (+ sparse se hybrid) → upsert PointStruct na física correta
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client.models import PointStruct

from server.src.fii_rag.chunking.base import DocContext, LoadedDocument
from server.src.fii_rag.chunking.registry import build_strategy
from server.src.fii_rag.embeddings.sparse import BM25SparseEmbedder
from server.src.fii_rag.extraction.chunk_metadata import ChunkMetadataExtractor
from server.src.fii_rag.extraction.doc_metadata import DocMetadataExtractor
from server.src.fii_rag.extraction.numerics import NumericFactsExtractor
from server.src.fii_rag.ingestion.enricher import ChunkEnricher
from server.src.fii_rag.ingestion.loader import PdfLoader
from server.src.fii_rag.schemas.chunk import ChunkPayload
from server.src.fii_rag.schemas.document import DocumentMetadata
from server.src.fii_rag.store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    CollectionNaming,
    QdrantRepository,
)
from server.src.fii_rag.store.provisioner import LogicalCollectionProvisioner

if TYPE_CHECKING:
    from server.src.fii_rag.config import AppConfig

DOC_ID_NAMESPACE = uuid5(NAMESPACE_DNS, "fii-rag-agent.doc-id")
DENSE_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 256


def compute_doc_id(source_filename: str, full_text: str) -> str:
    text_digest = hashlib.sha256(full_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return str(uuid5(DOC_ID_NAMESPACE, f"{source_filename}::{text_digest}"))


class IngestionPipeline:
    def __init__(
        self,
        config: "AppConfig",
        repository: QdrantRepository,
        provisioner: LogicalCollectionProvisioner,
        loader: PdfLoader,
        dense_embedder: Any,  # langchain Embeddings
        sparse_embedder: Optional[BM25SparseEmbedder],
        doc_extractor: DocMetadataExtractor,
        chunk_extractor: ChunkMetadataExtractor,
        numerics_extractor: NumericFactsExtractor,
    ):
        self.config = config
        self.repository = repository
        self.provisioner = provisioner
        self.loader = loader
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.doc_extractor = doc_extractor
        self.chunk_extractor = chunk_extractor
        self.numerics_extractor = numerics_extractor
        self.enricher = ChunkEnricher(chunk_extractor, numerics_extractor)

    def run(self, pdf_path: str, logical: str) -> dict:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(pdf_path)

        cfg = self.provisioner.read_logical_config(logical)
        if cfg is None:
            raise ValueError(
                f"Coleção lógica {logical!r} não existe. "
                "Crie-a no frontend antes de ingerir."
            )

        print(f"[IngestionPipeline] Carregando {path.name}...")
        loaded = self.loader.load(str(path))

        print(
            f"[IngestionPipeline] Extraindo metadados-doc via DOC_SUMMARY_LLM "
            f"({loaded.total_pages} páginas, {len(loaded.full_text)} chars)..."
        )
        extracted = self.doc_extractor.extract(loaded.full_text)
        doc_id = compute_doc_id(path.name, loaded.full_text)
        doc_meta = DocumentMetadata(
            doc_id=doc_id,
            source_filename=path.name,
            ingested_at=datetime.now(timezone.utc),
            total_pages=loaded.total_pages,
            **extracted.model_dump(),
        )
        ctx = DocContext(doc_id=doc_id, doc_metadata=doc_meta)

        strategies = list(cfg.get("strategies") or [])
        hybrid = bool(cfg.get("hybrid", True))
        print(
            f"[IngestionPipeline] Estratégias: {strategies}; hybrid={hybrid}; "
            f"doc_id={doc_id}; ticker={doc_meta.ticker}"
        )

        max_workers = max(1, min(self.config.ingestion.ingest_max_workers, len(strategies)))
        results: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self._process_strategy, name, loaded, ctx, logical, hybrid): name
                for name in strategies
            }
            for fut in futures:
                name = futures[fut]
                try:
                    n = fut.result()
                    results[name] = n
                    print(f"[IngestionPipeline] '{name}' → {n} pontos upsertados")
                except Exception as e:  # noqa: BLE001
                    print(f"[IngestionPipeline] '{name}' FALHOU: {e}")
                    results[name] = -1

        return {"doc_id": doc_id, "doc_meta": doc_meta.model_dump(), "per_strategy": results}

    def _process_strategy(
        self,
        name: str,
        loaded: LoadedDocument,
        ctx: DocContext,
        logical: str,
        hybrid: bool,
    ) -> int:
        strategy = build_strategy(name, self.config, embeddings=self.dense_embedder)
        chunks = strategy.chunk(loaded, ctx)
        if not chunks:
            return 0

        enriched = self.enricher.enrich(
            chunks,
            ctx,
            strategy_name=name,
            run_chunk_metadata=not strategy.produces_one_per_doc,
        )
        texts = [c.text for c in chunks]

        # Embed dense em batches
        dense_vectors: list[list[float]] = []
        for i in range(0, len(texts), DENSE_BATCH_SIZE):
            dense_vectors.extend(
                self.dense_embedder.embed_documents(texts[i : i + DENSE_BATCH_SIZE])
            )

        # Embed sparse se hybrid
        sparse_vectors = (
            self.sparse_embedder.embed_documents_to_qdrant(texts)
            if hybrid and self.sparse_embedder is not None
            else None
        )

        physical = CollectionNaming.to_physical(logical, name)
        points = []
        for i, (cm, dv) in enumerate(zip(enriched, dense_vectors)):
            payload = ChunkPayload.from_chunk_metadata(cm, text=texts[i]).model_dump()
            vector: dict = {DENSE_VECTOR_NAME: dv}
            if sparse_vectors is not None:
                vector[SPARSE_VECTOR_NAME] = sparse_vectors[i]
            points.append(PointStruct(id=cm.chunk_id, vector=vector, payload=payload))

        self.repository.upsert_points(physical, points, batch_size=UPSERT_BATCH_SIZE)
        return len(points)


def build_ingestion_pipeline(config: "AppConfig") -> IngestionPipeline:
    """Wiring centralizado — usado por `main.py` e `frontend/app.py`."""
    from server.src.fii_rag.embeddings import DenseEmbedderFactory, SparseEmbedderFactory
    from server.src.fii_rag.llm import LLMFactory, LLMStage
    from server.src.fii_rag.store import QdrantClientFactory

    client = QdrantClientFactory.get(config.qdrant_url)
    repository = QdrantRepository(client)
    provisioner = LogicalCollectionProvisioner(repository)

    dense = DenseEmbedderFactory(config).build_langchain()
    sparse = SparseEmbedderFactory(config).build()

    llm_factory = LLMFactory(config)
    doc_llm = llm_factory.for_stage(LLMStage.DOC_SUMMARY)
    chunk_llm = llm_factory.for_stage(LLMStage.CHUNK_METADATA)

    return IngestionPipeline(
        config=config,
        repository=repository,
        provisioner=provisioner,
        loader=PdfLoader(),
        dense_embedder=dense,
        sparse_embedder=sparse,
        doc_extractor=DocMetadataExtractor(doc_llm),
        chunk_extractor=ChunkMetadataExtractor(
            chunk_llm, max_workers=config.ingestion.ingest_max_workers * 2
        ),
        numerics_extractor=NumericFactsExtractor(llm=chunk_llm, use_llm_fallback=True),
    )
