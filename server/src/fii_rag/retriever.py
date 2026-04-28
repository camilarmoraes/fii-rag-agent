from typing import Any

from server.src.fii_rag.interfaces import IQueryEngineBuilder, IVectorStoreProvider

class HybridQueryEngineBuilder(IQueryEngineBuilder):
    """
    Constrói o Retriever com Re-ranking do LangChain.
    """
    def __init__(self, vector_store_provider: IVectorStoreProvider, top_k: int = 10, rerank_top_k: int = 3):
        self.vector_store_provider = vector_store_provider
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def build(self, collection_name: str = "fii_reports") -> Any:
        vector_store = self.vector_store_provider.get_store(collection_name)
        
        # Puxamos mais resultados inicialmente para filtrar depois -> (Similarity Reta)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})

        try:
            from langchain_classic.retrievers import ContextualCompressionRetriever
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            # O modelo BAAI é muito forte para reranking multi-lingual
            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            compressor = CrossEncoderReranker(model=model, top_n=self.rerank_top_k)

            # Combina a busca padrão de vetores do Qdrant com a camada densa Neural BAAI
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            return compression_retriever
        except ImportError:
            # Fallback caso não possuam sentence-transformers ativado
            print("Lembre de ter sentence-transformers instado para o Rerank poderoso funcionar.")
            return base_retriever
