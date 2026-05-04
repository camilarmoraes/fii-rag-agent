"""Embedder esparso BM25 via FastEmbed.

A saída de `embed_documents_to_qdrant`/`embed_query_to_qdrant` é diretamente
consumível como `vector["sparse"]` num `PointStruct` ou como `query` num
`Prefetch(using="sparse")` — sem mais conversões na borda.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdrant_client.models import SparseVector

from server.src.fii_rag.interfaces import ISparseEmbedder

if TYPE_CHECKING:
    from server.src.fii_rag.config import AppConfig


def _to_sparse_vector(raw: Any) -> SparseVector:
    """Converte um `fastembed.SparseEmbedding` em `qdrant_client.models.SparseVector`."""
    indices = raw.indices.tolist() if hasattr(raw.indices, "tolist") else list(raw.indices)
    values = raw.values.tolist() if hasattr(raw.values, "tolist") else list(raw.values)
    return SparseVector(indices=indices, values=values)


class BM25SparseEmbedder(ISparseEmbedder):
    """Embedder esparso BM25 via FastEmbed (modelo default `Qdrant/bm25`).

    Carrega o modelo lazy na primeira chamada para evitar download durante
    importação. Os métodos `*_to_qdrant` devolvem `SparseVector` prontos para
    o `qdrant-client`; os métodos sem sufixo retornam o objeto cru do
    FastEmbed (com atributos `indices`/`values` numpy).
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.model_name)

    # --- Saída crua (FastEmbed SparseEmbedding) -----------------------------
    def embed_documents(self, texts: list[str]) -> list[Any]:
        if not texts:
            return []
        self._ensure_loaded()
        return list(self._model.embed(texts))

    def embed_query(self, text: str) -> Any:
        self._ensure_loaded()
        return next(iter(self._model.query_embed([text])))

    # --- Saída pronta para o Qdrant -----------------------------------------
    def embed_documents_to_qdrant(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse_vector(r) for r in self.embed_documents(texts)]

    def embed_query_to_qdrant(self, text: str) -> SparseVector:
        return _to_sparse_vector(self.embed_query(text))


class SparseEmbedderFactory:
    """Fábrica de embedder esparso a partir do `AppConfig`.

    Hoje retorna sempre `BM25SparseEmbedder`. Para mudar para SPLADE, basta
    setar `SPARSE_EMBED_MODEL` para o id de um modelo SPLADE suportado pelo
    FastEmbed (ex: `prithivida/Splade_PP_en_v1`).
    """

    def __init__(self, config: "AppConfig"):
        self.config = config

    def build(self) -> BM25SparseEmbedder:
        return BM25SparseEmbedder(model_name=self.config.sparse_embed.model)
