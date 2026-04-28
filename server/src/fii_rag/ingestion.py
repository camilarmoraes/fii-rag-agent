import os
from langchain_community.document_loaders import PyMuPDFLoader

from server.src.fii_rag.interfaces import IVectorStoreProvider, IDocumentParser, IMetadataExtractor

class PDFIngestionManager:
    """
    Orquestra a ingestão de PDFs na base Qdrant através do LangChain.
    """
    
    def __init__(
        self, 
        vector_store_provider: IVectorStoreProvider,
        document_parser: IDocumentParser,
        metadata_extractor: IMetadataExtractor,
        data_dir: str = "data",
        collection_name: str = "fii_reports"
    ):
        self.vector_store_provider = vector_store_provider
        # LangChain text splitter
        self.document_parser = document_parser
        # Extrator de Metadados Semânticos
        self.metadata_extractor = metadata_extractor
        self.data_dir = data_dir
        self.collection_name = collection_name

    def run(self):
        print(f"Lendo documentos PDF do diretório: {self.data_dir}...")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Diretório {self.data_dir} criado. Adicione seus PDFs antes de rodar.")
            return

        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".pdf"):
                full_path = os.path.join(self.data_dir, file)
                print(f"Carregando: {full_path}")
                loader = PyMuPDFLoader(full_path)
                documents.extend(loader.load())

        if not documents:
            print("Nenhum PDF encontrado na pasta data/.")
            return

        vector_store = self.vector_store_provider.get_store(self.collection_name)
        
        parser = self.document_parser.get_parser()
        extractor_fn = self.metadata_extractor.get_extractors()

        print("Processando chunks (Recursive Character Split)...")
        # Transforma OS Documentos em Chunks (LangChain envia classes Document)
        splits = parser.split_documents(documents)
        
        # O extractor_fn é uma função que nós dovolvemos no chunking.py
        print(f"Iniciando API Mistral para extração semântica em {len(splits)} chunks... Isso pode levar um tempo.")
        enriched_splits = extractor_fn(splits)
        
        print(f"Inserindo {len(enriched_splits)} chunks gerados no Qdrant...")
        # Usa embeddings fornecidas no provider pelo LangChain nativamente
        vector_store.add_documents(documents=enriched_splits)
        
        print("Ingestão concluída com sucesso! (LangChain Ready)")
