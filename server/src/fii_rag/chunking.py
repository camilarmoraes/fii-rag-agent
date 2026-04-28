from typing import List, Any
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from server.src.fii_rag.interfaces import IDocumentParser, IMetadataExtractor

class LangChainParser(IDocumentParser):
    """
    Constrói a estrutura de Node Parse (Recursive Context Window).
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_parser(self) -> Any:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

# Schema do Pydantic para estruturar a saída do modelo
class ExtractedMetadata(BaseModel):
    title: str = Field(description="Título principal infériodo a partir do documento (FII).")
    keywords: List[str] = Field(description="5 ou mais palavras-chave fundamentais para a busca.")
    summary: str = Field(description="Resumo claro e descritivo focando em contexto de Fundo Imobiliário.")

class LangChainSemanticExtractor(IMetadataExtractor):
    """
    Monta a injeção semântica de Metadados processando os nós nativamente.
    Dessa forma, extraímos via Mistral os metadados antes de injetarmos no Qdrant.
    """
    def __init__(self, llm: Any):
        # Transforma o modelo para devolver sempre esse JSON com título, keywords e summary
        self.llm = llm.with_structured_output(ExtractedMetadata)

    def get_extractors(self) -> Any:
        def process_documents(documents: List[Document]) -> List[Document]:
            # Retorna uma função que processa a lista de splits
            # TODO: mudar a lógica de extração dos metadados
            total = len(documents)
            for i, doc in enumerate(documents):
                print(f"Extraindo metadados semânticos do Chunker {i+1}/{total} via Mistral...")
                try:
                    prompt = f"Analise o trecho do relatório de FII abaixo e extraia propriedades:\n\n{doc.page_content}"
                    metadata = self.llm.invoke(prompt)
                    # Mescla os metadados antigos (ex: paginação) com os novos via inteligência
                    doc.metadata["title"] = metadata.title
                    doc.metadata["keywords"] = metadata.keywords
                    doc.metadata["summary"] = metadata.summary
                except Exception as e:
                    print(f"Aviso: Erro ao extrair no chunk {i+1}: {e}")
            return documents
        
        return process_documents
