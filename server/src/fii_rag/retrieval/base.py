"""Tipos compartilhados pela camada de retrieval (multi-collection + multi-strategy).

`RetrievalResult` é o "documento" interno que circula entre `IRetriever` e os
adapters LangChain — carrega o `payload` cru do Qdrant para que o frontend e o
agent possam exibir badges (`strategy`, `ticker`, `report_year`, etc.) sem
re-querying.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    text: str
    score: float
    doc_id: str
    chunk_id: str
    strategy: str
    payload: dict = field(default_factory=dict)


class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...
