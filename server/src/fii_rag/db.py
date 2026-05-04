"""Shim de compatibilidade — `QdrantStoreProvider` agora delega ao `QdrantRepository`.

Mantém a superfície usada pelo `main.py` legado e pelo `frontend/app.py`
(`url`, `embed_model`, `get_store(...)`, `add_documents(...)`).

PR 3 acrescenta detecção automática de hybrid: se a coleção foi criada com
named vectors + sparse, o provider gera embedding sparse via BM25 e popula
`PointStruct.vector = {"dense": [...], "sparse": SparseVector(...)}`.

Será deletado no PR 6 junto com `IVectorStoreProvider`.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from qdrant_client.models import Distance, PointStruct

from server.src.fii_rag.embeddings import adapt_to_dimension
from server.src.fii_rag.embeddings.sparse import BM25SparseEmbedder
from server.src.fii_rag.interfaces import IVectorStoreProvider
from server.src.fii_rag.store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantClientFactory,
    QdrantRepository,
)

DEFAULT_COLLECTION = "fii_reports"


class QdrantStoreProvider(IVectorStoreProvider):
    """Adapter que conecta o pipeline legado ao `QdrantRepository`.

    Para hybrid search, mantém um `BM25SparseEmbedder` lazy. Se a coleção
    detectada não for hybrid, o sparse não é instanciado e o pipeline antigo
    (vetor unnamed) continua exatamente como antes.
    """

    def __init__(
        self,
        url: str,
        embed_model: Any,
        sparse_embedder: Optional[BM25SparseEmbedder] = None,
    ):
        self.url = url
        self.embed_model = embed_model
        self.client = QdrantClientFactory.get(url)
        self.repository = QdrantRepository(self.client)
        self._sparse_embedder = sparse_embedder
        self._active_collection: Optional[str] = None

    # ------------------------------------------------------------------
    # Sparse embedder (lazy)
    # ------------------------------------------------------------------

    def get_sparse_embedder(self) -> BM25SparseEmbedder:
        if self._sparse_embedder is None:
            self._sparse_embedder = BM25SparseEmbedder()
        return self._sparse_embedder

    # ------------------------------------------------------------------
    # IVectorStoreProvider — back-compat
    # ------------------------------------------------------------------

    def get_store(self, collection_name: str = DEFAULT_COLLECTION) -> "QdrantStoreProvider":
        """Garante que a coleção existe e adapta o embed_model à sua dimensão.

        Se a coleção ainda não existir, cria-a single-vector (sem hybrid).
        Hybrid é decidido pelo frontend no momento de criação da coleção via
        `repository.ensure_collection(..., hybrid=True)` — não aqui.
        """
        self._active_collection = collection_name
        if self.repository.collection_exists(collection_name):
            existing_dim = self.repository.detect_dense_dim(collection_name)
            if existing_dim:
                self.embed_model = adapt_to_dimension(self.embed_model, existing_dim)
                print(
                    f"[QdrantStoreProvider] Dimensão detectada: {existing_dim}d "
                    f"— embed_model reconfigurado."
                )
        else:
            probe = self.embed_model.embed_query("dimension probe")
            self.repository.ensure_collection(
                name=collection_name,
                dim=len(probe),
                distance=Distance.COSINE,
                hybrid=False,
            )
            print(
                f"[QdrantStoreProvider] Coleção '{collection_name}' criada com "
                f"dim={len(probe)}, distance=COSINE, hybrid=False."
            )
        return self

    # ------------------------------------------------------------------
    # Superfície que o PDFIngestionManager legado consome.
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Any]) -> None:
        """Embeda + upserta PointStructs nativos.

        Detecta automaticamente se a coleção é hybrid (sparse_vectors_config
        presente) e gera sparse embeddings via BM25 quando aplicável.
        """
        if not documents:
            return
        if self._active_collection is None:
            self.get_store(DEFAULT_COLLECTION)

        coll_name = self._active_collection
        is_hybrid = self.repository.has_sparse_vectors(coll_name)
        is_named = self.repository.is_named_vectors(coll_name)

        texts = [getattr(d, "page_content", str(d)) for d in documents]
        dense_vectors = self.embed_model.embed_documents(texts)

        sparse_vectors = (
            self.get_sparse_embedder().embed_documents_to_qdrant(texts)
            if is_hybrid
            else None
        )

        points: list[PointStruct] = []
        for idx, (doc, dense_vec, text) in enumerate(zip(documents, dense_vectors, texts)):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            payload = {"text": text, **metadata}

            vector: Any
            if is_hybrid:
                vector = {
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: sparse_vectors[idx],
                }
            elif is_named:
                vector = {DENSE_VECTOR_NAME: dense_vec}
            else:
                vector = dense_vec

            points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))

        self.repository.upsert_points(coll_name, points)
