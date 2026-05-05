"""Reciprocal Rank Fusion entre rankings de múltiplas estratégias.

Score final do ítem = Σ 1 / (k + rank_i) somando todas as posições nos
rankings em que ele aparece. Dedup por chave `(doc_id, page_number, sha256(text)[:12])`
— preserva o item com maior score original entre duplicatas.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from server.src.fii_rag.retrieval.base import RetrievalResult

DEFAULT_RRF_K = 60


def _dedup_key(item: RetrievalResult) -> str:
    page = item.payload.get("page_number") if item.payload else None
    digest = hashlib.sha256(item.text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{item.doc_id}::{page}::{digest}"


def rrf_merge(
    rankings: Iterable[list[RetrievalResult]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    rrf_scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievalResult] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = _dedup_key(item)
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in best or item.score > best[key].score:
                best[key] = item

    fused: list[RetrievalResult] = []
    for key, item in best.items():
        fused.append(
            RetrievalResult(
                text=item.text,
                score=rrf_scores[key],
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                strategy=item.strategy,
                payload=item.payload,
            )
        )

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused
