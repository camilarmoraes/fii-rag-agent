from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from server.src.fii_rag.llm.stages import LLMStage


# ============================================================================
# ABCs legadas — mantidas para compatibilidade com db.py / chunking.py /
# retriever.py / ingestion.py enquanto a refatoração progride.
# Serão removidas no PR 6.
# ============================================================================

class IVectorStoreProvider(ABC):
    """Contrato para provimento de acesso a bancos vetoriais."""

    @abstractmethod
    def get_store(self, collection_name: str) -> Any:
        pass


class IDocumentParser(ABC):
    """Contrato para regras de particionamento (Chunking) de documentos."""

    @abstractmethod
    def get_parser(self) -> Any:
        pass


class IMetadataExtractor(ABC):
    """Contrato para extrações de features de metadados em nós."""

    @abstractmethod
    def get_extractors(self) -> List[Any]:
        pass


class IQueryEngineBuilder(ABC):
    """Contrato para a construção do motor de busca a partir de um vector store."""

    @abstractmethod
    def build(self, collection_name: str) -> Any:
        pass


# ============================================================================
# ABCs novas — base da arquitetura Qdrant nativo + multi-strategy + two-stage.
# ============================================================================

class IDenseEmbedder(ABC):
    """Embedder denso (vetor único por texto)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...


class ISparseEmbedder(ABC):
    """Embedder esparso (BM25/SPLADE) — saída compatível com `SparseVector`."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Any]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> Any:
        ...


class ILLMFactory(ABC):
    """Fábrica de `BaseChatModel` por estágio do pipeline."""

    @abstractmethod
    def for_stage(self, stage: "LLMStage") -> "BaseChatModel":
        ...
