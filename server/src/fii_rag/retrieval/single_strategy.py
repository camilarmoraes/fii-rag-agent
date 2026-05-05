"""Retriever simples sobre 1 coleção (sub-strategy isolada ou coll legacy).

Usa `query_hybrid` se a coleção tem `sparse_vectors`; senão `query_dense`.
Por default aplica `exclude_config_filter()` para esconder o ponto
`_LOGICAL_CONFIG_` em colls lógicas; em colls legacy passa `with_config_filter=False`.
"""

from __future__ import annotations

from typing import Any, Optional

from server.src.fii_rag.retrieval.base import IRetriever, RetrievalResult
from server.src.fii_rag.store import (
    DENSE_VECTOR_NAME,
    QdrantRepository,
    exclude_config_filter,
)


def point_to_result(point: Any) -> RetrievalResult:
    payload = dict(point.payload or {})
    text = payload.get("text") or payload.get("page_content") or ""
    return RetrievalResult(
        text=text,
        score=float(point.score) if getattr(point, "score", None) is not None else 0.0,
        doc_id=payload.get("doc_id") or "",
        chunk_id=str(point.id),
        strategy=payload.get("strategy") or "unknown",
        payload=payload,
    )


class SingleStrategyRetriever(IRetriever):
    def __init__(
        self,
        repository: QdrantRepository,
        collection_name: str,
        dense_embedder: Any,
        sparse_embedder: Optional[Any] = None,
        with_config_filter: bool = True,
    ):
        self.repo = repository
        self.collection = collection_name
        self.dense = dense_embedder
        self.sparse = sparse_embedder
        self.with_config_filter = with_config_filter

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        q_dense = self.dense.embed_query(query)
        flt = exclude_config_filter() if self.with_config_filter else None

        if self.repo.has_sparse_vectors(self.collection) and self.sparse is not None:
            q_sparse = self.sparse.embed_query_to_qdrant(query)
            scored = self.repo.query_hybrid(
                self.collection,
                dense_vector=q_dense,
                sparse_vector=q_sparse,
                limit=top_k,
                query_filter=flt,
            )
        else:
            using = (
                DENSE_VECTOR_NAME if self.repo.is_named_vectors(self.collection) else None
            )
            scored = self.repo.query_dense(
                self.collection,
                vector=q_dense,
                using=using,
                limit=top_k,
                query_filter=flt,
            )

        return [point_to_result(p) for p in scored]
