"""Wrappers LangChain legados — preservados para o `PDFIngestionManager` antigo.

Nova arquitetura usa `chunking/recursive.py` + `extraction/chunk_metadata.py`.
Estes shims serão removidos no PR 6 junto com o pipeline legado.
"""

from typing import Any, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from server.src.fii_rag.interfaces import IDocumentParser, IMetadataExtractor


class LangChainParser(IDocumentParser):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_parser(self) -> Any:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )


class ExtractedMetadata(BaseModel):
    title: str = Field(description="Título principal infériodo a partir do documento (FII).")
    keywords: List[str] = Field(description="5 ou mais palavras-chave fundamentais para a busca.")
    summary: str = Field(description="Resumo claro e descritivo focando em contexto de Fundo Imobiliário.")


class LangChainSemanticExtractor(IMetadataExtractor):
    def __init__(self, llm: Any):
        self.llm = llm.with_structured_output(ExtractedMetadata)

    def get_extractors(self) -> Any:
        def process_documents(documents: List[Document]) -> List[Document]:
            total = len(documents)
            for i, doc in enumerate(documents):
                print(f"Extraindo metadados semânticos do Chunker {i + 1}/{total} via Mistral...")
                try:
                    prompt = (
                        "Analise o trecho do relatório de FII abaixo e extraia propriedades:\n\n"
                        f"{doc.page_content}"
                    )
                    metadata = self.llm.invoke(prompt)
                    doc.metadata["title"] = metadata.title
                    doc.metadata["keywords"] = metadata.keywords
                    doc.metadata["summary"] = metadata.summary
                except Exception as e:
                    print(f"Aviso: Erro ao extrair no chunk {i + 1}: {e}")
            return documents

        return process_documents
