"""Retriever híbrido (vetor denso + cross-encoder rerank) sobre Qdrant nativo.

Substitui o uso de `langchain_classic.retrievers.ContextualCompressionRetriever`
+ `langchain_community.cross_encoders.HuggingFaceCrossEncoder`. A engine de
busca é o `QdrantRepository` puro; o reranker é o `CrossEncoderReranker`
baseado em `sentence-transformers` direto.

Para preservar a integração com `langchain_classic.chains.create_retrieval_chain`
(usada pelo `agent.py`), expomos a engine via uma subclasse de
`langchain_core.retrievers.BaseRetriever` que devolve `Document`s.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from server.src.fii_rag.db import QdrantStoreProvider
from server.src.fii_rag.interfaces import IQueryEngineBuilder
from server.src.fii_rag.retrieval.rerank import CrossEncoderReranker


def _payload_to_document(payload: dict) -> Document:
    """Adapta o payload de um `ScoredPoint` para `langchain_core.Document`.

    Aceita tanto `text` (formato novo, usado pelo PR 2 em diante) quanto
    `page_content` (formato antigo do `langchain-qdrant`) na mesma coleção.
    """
    text = payload.get("text") or payload.get("page_content") or ""
    metadata = {k: v for k, v in payload.items() if k not in ("text", "page_content")}
    return Document(page_content=text, metadata=metadata)


class _NativeQdrantRetriever(BaseRetriever):
    """`BaseRetriever` que delega ao `QdrantRepository` + reranker local."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: QdrantStoreProvider
    collection_name: str
    top_k: int = 10
    rerank_top_k: int = 3
    reranker: Optional[CrossEncoderReranker] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        embed_model = self.provider.embed_model
        repo = self.provider.repository
        query_dense = embed_model.embed_query(query)

        if repo.has_sparse_vectors(self.collection_name):
            query_sparse = self.provider.get_sparse_embedder().embed_query_to_qdrant(query)
            scored = repo.query_hybrid(
                self.collection_name,
                dense_vector=query_dense,
                sparse_vector=query_sparse,
                limit=self.top_k,
            )
        else:
            using = "dense" if repo.is_named_vectors(self.collection_name) else None
            scored = repo.query_dense(
                self.collection_name,
                vector=query_dense,
                using=using,
                limit=self.top_k,
            )

        docs = [_payload_to_document(p.payload or {}) for p in scored]

        if self.reranker is not None and len(docs) > self.rerank_top_k:
            scores = self.reranker.score(query, [d.page_content for d in docs])
            ranked = sorted(zip(scores, docs), key=lambda p: p[0], reverse=True)
            docs = [d for _, d in ranked[: self.rerank_top_k]]
        else:
            docs = docs[: self.rerank_top_k]
        return docs


class HybridQueryEngineBuilder(IQueryEngineBuilder):
    """Constrói um `BaseRetriever` que faz vector top-K + rerank top-N.

    O nome "Hybrid" é histórico — sparse vectors entram no PR 3.
    """

    def __init__(
        self,
        vector_store_provider: QdrantStoreProvider,
        top_k: int = 10,
        rerank_top_k: int = 3,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.vector_store_provider = vector_store_provider
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.reranker_model = reranker_model

    def build(self, collection_name: str = "fii_reports") -> Any:
        # Garante que o embed_model está adaptado à dim da coleção.
        self.vector_store_provider.get_store(collection_name)

        try:
            reranker: Optional[CrossEncoderReranker] = CrossEncoderReranker(
                model_name=self.reranker_model
            )
        except Exception as e:  # noqa: BLE001
            print(
                "Aviso: não foi possível inicializar o CrossEncoderReranker — "
                "caindo para retriever sem rerank. "
                f"Verifique se 'sentence-transformers' está instalado. ({e})"
            )
            reranker = None

        return _NativeQdrantRetriever(
            provider=self.vector_store_provider,
            collection_name=collection_name,
            top_k=self.top_k,
            rerank_top_k=self.rerank_top_k,
            reranker=reranker,
        )
