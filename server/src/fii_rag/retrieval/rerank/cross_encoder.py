"""Reranker via `sentence_transformers.CrossEncoder` direto, sem LangChain.

Substitui o combo `langchain_classic.retrievers.document_compressors.CrossEncoderReranker`
+ `langchain_community.cross_encoders.HuggingFaceCrossEncoder` usado pel
`HybridQueryEngineBuilder` legado.
"""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._ensure_loaded()
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]
