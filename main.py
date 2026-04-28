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

    # import os
    # from dotenv import load_dotenv
    # from mistralai.client import Mistral

    # load_dotenv()
    # client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    #     # Com dimensão reduzida e precisão int8 (mais econômico)
    # response = client.embeddings.create(
    #     model="codestral-embed-2505",
    #     inputs=["SELECT * FROM users WHERE id = ?"],
    #     output_dtype="int8",       # float | int8 | uint8 | binary | ubinary
    #     output_dimension=512,      # padrão: 1536, máximo: 3072
    # )

    # print(f"Dimensão: {len(response.data[0].embedding)}")
    # print(response.data[0].embedding)

    from langchain_mistralai import MistralAIEmbeddings

    embeddings = MistralAIEmbeddings(
        model="codestral-embed-2505",
    )
    text = "Text qualquer"
    single_vector = embeddings.embed_query(text)
    print(single_vector)

if __name__ == '__main__':
    main()
