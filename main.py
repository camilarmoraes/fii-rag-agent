import argparse
from server.src.fii_rag.config import AppConfig
from server.src.fii_rag.db import QdrantStoreProvider
from server.src.fii_rag.chunking import LangChainParser, LangChainSemanticExtractor
from server.src.fii_rag.ingestion import PDFIngestionManager
from server.src.fii_rag.retriever import HybridQueryEngineBuilder
from server.src.fii_rag.agent import RAGAgent

def setup_di_container():
    """
    Simula um Container de Injeção de Dependências.
    No futuro, usando Flask, isso pode ser instanciado dentro de Application Factories.
    """
    config = AppConfig()
    llm, embed_model = config.get_llm_and_embeddings()
    
    # Provider do BD com embeddings LangChain nativos
    qdrant_provider = QdrantStoreProvider(url=config.qdrant_url, embed_model=embed_model)
    parser = LangChainParser()
    extractor = LangChainSemanticExtractor(llm=llm)
    
    # Manager para inserção contendo as injetadas
    ingestion_manager = PDFIngestionManager(
        vector_store_provider=qdrant_provider,
        document_parser=parser,
        metadata_extractor=extractor
    )
    
    # Builder para busca
    query_builder = HybridQueryEngineBuilder(
        vector_store_provider=qdrant_provider,
        top_k=10,
        rerank_top_k=3
    )
    
    # Agente interativo completo LCEL
    agent = RAGAgent(query_engine_builder=query_builder, llm=llm)
    
    return ingestion_manager, agent

def main():
    parser = argparse.ArgumentParser(description="FII RAG Agent CLI (SOLID)")
    parser.add_argument("--ingest", action="store_true", help="Roda o script de ingestão e parseia os PDFs em data/")
    parser.add_argument("--chat", action="store_true", help="Inicia o servidor de chat / QA no terminal")

    args = parser.parse_args()

    # Prepara o escopo global resolvendo dependências
    ingestion_manager, agent = setup_di_container()

    if args.ingest:
        ingestion_manager.run()
    elif args.chat:
        agent.chat_loop()
    else:
        print("Uso padrão: \nPara processar PDFs: python main.py --ingest\nPara interagir: python main.py --chat")

if __name__ == '__main__':
    main()
