"""Shim de compatibilidade — `QdrantStoreProvider` agora delega ao `QdrantRepository`.

Mantém a superfície usada pelo `main.py` legado e pelo `frontend/app.py`
(`url`, `embed_model`, `get_store(...)`). A diferença é que `get_store` não
retorna mais um `QdrantVectorStore` (langchain-qdrant) — retorna o próprio
`provider`, que expõe `add_documents` consumido pelo `PDFIngestionManager`
reescrito.

Será deletado no PR 6 junto com a interface `IVectorStoreProvider`.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from qdrant_client.models import Distance, PointStruct

from server.src.fii_rag.embeddings import adapt_to_dimension
from server.src.fii_rag.interfaces import IVectorStoreProvider
from server.src.fii_rag.store import QdrantClientFactory, QdrantRepository

DEFAULT_COLLECTION = "fii_reports"


class QdrantStoreProvider(IVectorStoreProvider):
    """Adapter que conecta o pipeline legado ao `QdrantRepository`.

    Antes: encapsulava `langchain_qdrant.QdrantVectorStore`.
    Agora: encapsula `QdrantRepository` + faz upsert de PointStruct nativo.

    O método `get_store` retorna `self`; o caller usa `.add_documents(docs)`
    diretamente. Em produção nova (PR 4+), os consumidores devem usar
    `provider.repository` em vez do shim.
    """

    def __init__(self, url: str, embed_model: Any):
        self.url = url
        self.embed_model = embed_model
        self.client = QdrantClientFactory.get(url)
        self.repository = QdrantRepository(self.client)
        self._active_collection: Optional[str] = None

    # ------------------------------------------------------------------
    # IVectorStoreProvider — back-compat
    # ------------------------------------------------------------------

    def get_store(self, collection_name: str = DEFAULT_COLLECTION) -> "QdrantStoreProvider":
        """Garante que a coleção existe e adapta o embed_model à sua dimensão.

        Retorna `self` para permitir o uso ergonômico:
            store = provider.get_store("fii_reports")
            store.add_documents(docs)
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
                f"dim={len(probe)}, distance=COSINE."
            )
        return self

    # ------------------------------------------------------------------
    # Superfície que o PDFIngestionManager legado consome.
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Any]) -> None:
        """Embeda + upserta PointStructs nativos.

        Aceita `langchain_core.documents.Document` (que é o que sai do
        LangChainParser/LangChainSemanticExtractor) — usa `page_content` e
        `metadata`. O payload Qdrant fica `{"text": page_content, **metadata}`.
        """
        if not documents:
            return
        if self._active_collection is None:
            self.get_store(DEFAULT_COLLECTION)

        texts = [getattr(d, "page_content", str(d)) for d in documents]
        vectors = self.embed_model.embed_documents(texts)

        points: list[PointStruct] = []
        for doc, vec, text in zip(documents, vectors, texts):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            payload = {"text": text, **metadata}
            points.append(PointStruct(id=str(uuid4()), vector=vec, payload=payload))

        self.repository.upsert_points(self._active_collection, points)
