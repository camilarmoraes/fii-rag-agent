from abc import ABC, abstractmethod
from typing import Any, List

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
