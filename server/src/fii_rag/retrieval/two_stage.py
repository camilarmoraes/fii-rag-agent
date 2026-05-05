"""Two-stage retriever sobre uma coleção lógica multi-strategy.

Stage 1: busca em `<logical>__summary` para identificar os `doc_id`s mais
relevantes (apenas dense — 1 ponto por documento).

Stage 2: busca paralela nas demais sub-collections com `Filter(doc_id IN [...])`,
hybrid (dense + sparse) quando ativo. Funde os rankings via RRF e opcionalmente
re-rankeia o top-N final via cross-encoder ou LLM.

Hidratação hierarchical: quando o resultado vem da estratégia `hierarchical`,
o `parent_text` (guardado inline no payload do filho) é prependido ao texto
final para dar contexto largo ao LLM.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
)

from server.src.fii_rag.retrieval.base import IRetriever, RetrievalResult
from server.src.fii_rag.retrieval.fusion import DEFAULT_RRF_K, rrf_merge
from server.src.fii_rag.retrieval.single_strategy import point_to_result
from server.src.fii_rag.store import (
    DENSE_VECTOR_NAME,
    CollectionNaming,
    QdrantRepository,
    exclude_config_filter,
)

NON_SUMMARY_STRATEGIES = ("recursive", "semantic", "doc_aware", "hierarchical")


class TwoStageRetriever(IRetriever):
    def __init__(
        self,
        repository: QdrantRepository,
        logical: str,
        strategies: list[str],
        hybrid: bool,
        dense_embedder: Any,
        sparse_embedder: Optional[Any] = None,
        stage1_top_k: int = 8,
        stage2_top_k_per_strategy: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
        rerank_top_n: int = 30,
        reranker: Optional[Any] = None,
        max_workers: int = 4,
    ):
        self.repo = repository
        self.logical = logical
        # Apenas as não-summary entram no stage 2
        self.stage2_strategies = [
            s for s in strategies if s in NON_SUMMARY_STRATEGIES
        ]
        self.hybrid = hybrid
        self.dense = dense_embedder
        self.sparse = sparse_embedder
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k_per_strategy = stage2_top_k_per_strategy
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.reranker = reranker
        self.max_workers = max(1, min(max_workers, max(len(self.stage2_strategies), 1)))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        q_dense = self.dense.embed_query(query)
        q_sparse = (
            self.sparse.embed_query_to_qdrant(query)
            if self.hybrid and self.sparse is not None
            else None
        )

        # Stage 1 — summary → doc_ids
        summary_coll = CollectionNaming.to_physical(self.logical, "summary")
        try:
            scored = self.repo.query_dense(
                summary_coll,
                vector=q_dense,
                using=DENSE_VECTOR_NAME,
                limit=self.stage1_top_k,
                query_filter=exclude_config_filter(),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[TwoStageRetriever] Stage 1 falhou em {summary_coll}: {e}")
            return []

        doc_ids = [
            p.payload["doc_id"]
            for p in scored
            if p.payload and p.payload.get("doc_id")
        ]
        print(
            f"[TwoStageRetriever] Stage 1: {len(doc_ids)} doc_ids "
            f"({doc_ids[:3]}{'...' if len(doc_ids) > 3 else ''})"
        )

        # Sem doc_ids: fallback retorna os summaries diretamente
        if not doc_ids:
            return [point_to_result(p) for p in scored][:top_k]

        # Stage 2 — paralelo por strategy
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
        )
        rankings: list[list[RetrievalResult]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(
                    self._query_strategy, name, q_dense, q_sparse, doc_filter
                ): name
                for name in self.stage2_strategies
            }
            for fut in futures:
                name = futures[fut]
                try:
                    res = fut.result()
                    rankings.append(res)
                    print(f"[TwoStageRetriever] Stage 2 '{name}': {len(res)} candidates")
                except Exception as e:  # noqa: BLE001
                    print(f"[TwoStageRetriever] Stage 2 '{name}' falhou: {e}")

        if not rankings:
            return []

        # Fusão RRF + dedup
        fused = rrf_merge(rankings, k=self.rrf_k)
        # Hidratação hierarchical
        for r in fused:
            if r.strategy == "hierarchical":
                parent_text = (r.payload or {}).get("parent_text")
                if parent_text and parent_text not in r.text:
                    r.text = f"{parent_text}\n\n{r.text}"

        # Rerank opcional
        if self.reranker is not None and fused:
            candidates = fused[: self.rerank_top_n]
            scores = self.reranker.score(query, [c.text for c in candidates])
            ranked = sorted(
                zip(scores, candidates), key=lambda p: p[0], reverse=True
            )
            return [c for _, c in ranked[:top_k]]

        return fused[:top_k]

    def _query_strategy(
        self,
        name: str,
        q_dense: list[float],
        q_sparse: Any,
        doc_filter: Filter,
    ) -> list[RetrievalResult]:
        physical = CollectionNaming.to_physical(self.logical, name)
        if self.hybrid and q_sparse is not None:
            scored = self.repo.query_hybrid(
                physical,
                dense_vector=q_dense,
                sparse_vector=q_sparse,
                limit=self.stage2_top_k_per_strategy,
                query_filter=doc_filter,
            )
        else:
            scored = self.repo.query_dense(
                physical,
                vector=q_dense,
                using=DENSE_VECTOR_NAME,
                limit=self.stage2_top_k_per_strategy,
                query_filter=doc_filter,
            )
        return [point_to_result(p) for p in scored]
