"""Adapter LangChain — embrulha um `IRetriever` num `BaseRetriever`.

Necessário enquanto `agent.py` ainda usa `langchain_classic.chains.create_retrieval_chain`,
que espera um `BaseRetriever` retornando `Document`s. No PR 6 esta camada
desaparece quando a chain do agent for reescrita à mão.

A metadata do `Document` carrega o payload do Qdrant + 3 chaves prefixadas
com `_` (`_score`, `_strategy`, `_chunk_id`) para o frontend exibir badges.
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from server.src.fii_rag.retrieval.base import IRetriever


class LangChainRetrieverAdapter(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    irretriever: Any  # IRetriever — Any para evitar erro do pydantic com ABC
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.irretriever.retrieve(query, top_k=self.top_k)
        docs: list[Document] = []
        for r in results:
            metadata = dict(r.payload or {})
            metadata["_score"] = r.score
            metadata["_strategy"] = r.strategy
            metadata["_chunk_id"] = r.chunk_id
            docs.append(Document(page_content=r.text, metadata=metadata))
        return docs
