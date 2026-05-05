"""`RetrieverBuilder` — orquestra a construção de IRetrievers.

Lê o `_LOGICAL_CONFIG_` da coleção lógica para decidir entre `TwoStageRetriever`
(lógica) e `SingleStrategyRetriever` (legacy). Também resolve o reranker
adequado a partir do `RERANK_BACKEND` no `.env`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from server.src.fii_rag.retrieval.rerank import CrossEncoderReranker, LLMReranker
from server.src.fii_rag.retrieval.single_strategy import SingleStrategyRetriever
from server.src.fii_rag.retrieval.two_stage import TwoStageRetriever
from server.src.fii_rag.store import (
    LogicalCollectionProvisioner,
    QdrantRepository,
)

if TYPE_CHECKING:
    from server.src.fii_rag.config import AppConfig


class RetrieverBuilder:
    def __init__(
        self,
        config: "AppConfig",
        repository: QdrantRepository,
        provisioner: LogicalCollectionProvisioner,
        dense_embedder: Any,
        sparse_embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
    ):
        self.config = config
        self.repo = repository
        self.provisioner = provisioner
        self.dense = dense_embedder
        self.sparse = sparse_embedder
        # Lazy: se não passar reranker, resolve via config no primeiro uso
        self._reranker = reranker
        self._reranker_resolved = reranker is not None

    def get_reranker(self) -> Optional[Any]:
        if self._reranker_resolved:
            return self._reranker
        backend = self.config.retrieval.rerank_backend.lower()
        try:
            if backend == "cross_encoder":
                self._reranker = CrossEncoderReranker(
                    model_name=self.config.retrieval.rerank_cross_encoder_model
                )
            elif backend == "llm":
                from server.src.fii_rag.llm import LLMFactory, LLMStage

                rerank_llm = LLMFactory(self.config).for_stage(LLMStage.RERANK)
                self._reranker = LLMReranker(llm=rerank_llm)
            else:
                print(f"[RetrieverBuilder] RERANK_BACKEND desconhecido: {backend!r}.")
                self._reranker = None
        except Exception as e:  # noqa: BLE001
            print(f"[RetrieverBuilder] Falhou ao instanciar reranker ({e}). Sem rerank.")
            self._reranker = None
        self._reranker_resolved = True
        return self._reranker

    def build_for_logical(self, logical: str) -> TwoStageRetriever:
        cfg = self.provisioner.read_logical_config(logical)
        if cfg is None:
            raise ValueError(
                f"Coleção lógica {logical!r} não tem `_LOGICAL_CONFIG_`. "
                "Crie-a pelo frontend antes."
            )
        return TwoStageRetriever(
            repository=self.repo,
            logical=logical,
            strategies=list(cfg.get("strategies", ["recursive"])),
            hybrid=bool(cfg.get("hybrid", True)),
            dense_embedder=self.dense,
            sparse_embedder=self.sparse,
            stage1_top_k=self.config.retrieval.stage1_top_k,
            stage2_top_k_per_strategy=self.config.retrieval.stage2_top_k_per_strategy,
            rrf_k=self.config.retrieval.rrf_k,
            reranker=self.get_reranker(),
        )

    def build_for_legacy(self, name: str) -> SingleStrategyRetriever:
        return SingleStrategyRetriever(
            repository=self.repo,
            collection_name=name,
            dense_embedder=self.dense,
            sparse_embedder=self.sparse,
            with_config_filter=False,
        )


def build_retriever_builder(config: "AppConfig") -> RetrieverBuilder:
    """Wiring centralizado para construir `RetrieverBuilder` a partir do config."""
    from server.src.fii_rag.embeddings import (
        DenseEmbedderFactory,
        SparseEmbedderFactory,
    )
    from server.src.fii_rag.store import QdrantClientFactory

    client = QdrantClientFactory.get(config.qdrant_url)
    repo = QdrantRepository(client)
    return RetrieverBuilder(
        config=config,
        repository=repo,
        provisioner=LogicalCollectionProvisioner(repo),
        dense_embedder=DenseEmbedderFactory(config).build_langchain(),
        sparse_embedder=SparseEmbedderFactory(config).build(),
    )
