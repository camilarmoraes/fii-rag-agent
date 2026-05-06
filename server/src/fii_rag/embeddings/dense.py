from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from server.src.fii_rag.config import AppConfig


def adapt_to_dimension(embedder: Any, dim: int) -> Any:
    """Reconfigura um embedder LangChain para gerar vetores de dimensão `dim`.

    Suporta:
    - `GoogleGenerativeAIEmbeddings` → `output_dimensionality`
    - `MistralAIEmbeddings`         → `output_dimension` (se a versão expuser o campo)

    Para outros embedders, retorna o original sem alteração — útil quando a
    coleção pré-existente foi criada com a dimensão nativa do modelo.
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if isinstance(embedder, GoogleGenerativeAIEmbeddings):
            return embedder.model_copy(update={"output_dimensionality": dim})
    except ImportError:
        pass

    try:
        from langchain_mistralai import MistralAIEmbeddings

        if isinstance(embedder, MistralAIEmbeddings):
            if "output_dimension" in MistralAIEmbeddings.model_fields:
                return embedder.model_copy(update={"output_dimension": dim})
    except ImportError:
        pass

    return embedder


class DenseEmbedderFactory:
    """Constrói um embedder denso a partir da config.

    Por enquanto retorna o objeto LangChain `Embeddings` diretamente
    (back-compat com o código legado). No PR 2 será introduzido um wrapper
    `IDenseEmbedder` que esconde o vendor.
    """

    def __init__(self, config: "AppConfig"):
        self.config = config

    def build_langchain(self) -> Any:
        cfg = self.config.dense_embed
        provider = cfg.provider.lower()

        if provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=cfg.model,
                google_api_key=self.config.api_keys.google_api_key,
            )
        if provider == "mistral":
            from langchain_mistralai import MistralAIEmbeddings

            return MistralAIEmbeddings(
                model=cfg.model,
                mistral_api_key=self.config.api_keys.mistral_api_key,
            )
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=cfg.model,
                openai_api_key=self.config.api_keys.openai_api_key,
            )

        raise ValueError(
            f"Provider de embeddings desconhecido: {cfg.provider!r}. "
            f"Suportados: google, mistral, openai."
        )
