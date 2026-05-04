from typing import TYPE_CHECKING

from server.src.fii_rag.interfaces import ILLMFactory
from server.src.fii_rag.llm.stages import LLMStage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from server.src.fii_rag.config import AppConfig


class LLMFactory(ILLMFactory):
    """Constrói um `BaseChatModel` para um estágio específico, lendo a config.

    Cada estágio tem um provider e modelo independentes definidos no `.env`.
    Suporta providers: `mistral`, `google`, `openai`.
    """

    def __init__(self, config: "AppConfig"):
        self.config = config

    def for_stage(self, stage: LLMStage) -> "BaseChatModel":
        cfg = self.config.llm_stages[stage]
        provider = cfg.provider.lower()

        if provider == "mistral":
            from langchain_mistralai import ChatMistralAI

            return ChatMistralAI(
                model=cfg.model,
                mistral_api_key=self.config.api_keys.mistral_api_key,
                temperature=cfg.temperature,
            )
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=cfg.model,
                google_api_key=self.config.api_keys.google_api_key,
                temperature=cfg.temperature,
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=cfg.model,
                openai_api_key=self.config.api_keys.openai_api_key,
                temperature=cfg.temperature,
            )

        raise ValueError(
            f"Provider LLM desconhecido para o estágio {stage.value}: {cfg.provider!r}. "
            f"Suportados: mistral, google, openai."
        )
