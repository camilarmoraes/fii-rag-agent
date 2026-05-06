from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from server.src.fii_rag.config import AppConfig


class BM25SparseEmbedder:
    """Embedder esparso BM25 via FastEmbed.

    Carrega o modelo lazy (na primeira chamada de `embed_*`) para evitar baixar
    pesos durante a importação. A saída é compatível com
    `qdrant_client.models.SparseVector` — cada item tem atributos `indices` e
    `values`.

    Esqueleto do PR 1; a integração com `IngestionPipeline` e
    `TwoStageRetriever` chega no PR 3.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.model_name)

    def embed_documents(self, texts: list[str]) -> list[Any]:
        self._ensure_loaded()
        return list(self._model.embed(texts))

    def embed_query(self, text: str) -> Any:
        self._ensure_loaded()
        return next(iter(self._model.query_embed([text])))


class SparseEmbedderFactory:
    """Constrói um embedder esparso a partir da config.

    Hoje retorna sempre `BM25SparseEmbedder`. Caso o usuário queira SPLADE no
    futuro, basta trocar o `SPARSE_EMBED_MODEL` para um identificador SPLADE
    suportado pelo FastEmbed.
    """

    def __init__(self, config: "AppConfig"):
        self.config = config

    def build(self) -> BM25SparseEmbedder:
        return BM25SparseEmbedder(model_name=self.config.sparse_embed.model)
