from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from langchain_qdrant import QdrantVectorStore

from server.src.fii_rag.interfaces import IVectorStoreProvider

class QdrantStoreProvider(IVectorStoreProvider):
    """
    # TODO: Refatorar para usar o strategy pattern?
    Provedor do banco Qdrant pelo LangChain.
    Centraliza a lógica de conexão e acesso ao vector store.
    """
    def __init__(self, url: str, embed_model):
        self.url = url
        self.embed_model = embed_model
        # Qdrant Vector Store requer o embed_model para criar embeddings no background,
        # bem como setup para Sparse Vectors se configurado nativamente.
        self._client = QdrantClient(url=self.url)

    def _adapt_embed_model(self, dimension: int):
        """
        Retorna uma cópia do embed_model reconfigurada para a dimensão
        da collection existente no Qdrant.

        Modelos suportados:  # TODO: adicionar método genérico para suportar outros modelos e provedores
        - GoogleGenerativeAIEmbeddings → parâmetro: output_dimensionality
        - MistralAIEmbeddings          → parâmetro: output_dimension (se disponível na versão instalada)
        """
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if isinstance(self.embed_model, GoogleGenerativeAIEmbeddings):
                return self.embed_model.model_copy(
                    update={"output_dimensionality": dimension}
                )
        except ImportError:
            pass

        try:
            from langchain_mistralai import MistralAIEmbeddings
            if isinstance(self.embed_model, MistralAIEmbeddings):
                # Verifica se a versão instalada expõe o campo output_dimension
                if "output_dimension" in MistralAIEmbeddings.model_fields:
                    return self.embed_model.model_copy(
                        update={"output_dimension": dimension}
                    )
                else:
                    print(
                        f"[QdrantStoreProvider] Aviso: MistralAIEmbeddings na versão instalada "
                        f"não expõe 'output_dimension'. Certifique-se de criar a collection com "
                        f"a dimensão nativa do modelo."
                    )
        except ImportError:
            pass

        # Fallback: retorna o modelo original sem alteração
        return self.embed_model


    def get_store(self, collection_name: str = "fii_reports") -> QdrantVectorStore:
        """
        Retorna o QdrantVectorStore para a collection indicada.
        Detecta automaticamente a dimensão vetorial configurada na collection
        e reconfigura o embed_model para gerar vetores compatíveis.
        """
        embed_model = self.embed_model
        try:
            info = self._client.get_collection(collection_name)
            cfg = info.config.params.vectors
            # Suporta collections com vetor único e named vectors
            if isinstance(cfg, dict):
                vec_dim = next(iter(cfg.values())).size
            else:
                vec_dim = getattr(cfg, "size", None)

            if vec_dim:
                embed_model = self._adapt_embed_model(vec_dim)
                print(f"[QdrantStoreProvider] Dimensão detectada: {vec_dim}d — embed_model reconfigurado.")
        except Exception as e:
            print(f"[QdrantStoreProvider] Aviso: não foi possível detectar dimensão da collection '{collection_name}': {e}")

        return QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=embed_model,
        )
