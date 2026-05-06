"""Configuração agregada do app, lida de `.env` via `pydantic-settings`.

A classe `AppConfig` mantém compatibilidade com o código legado:
    - `config.qdrant_url` continua funcionando (proxy para `config.qdrant.qdrant_url`)
    - `config.get_llm_and_embeddings()` continua retornando `(chat_llm, embeddings)`
      construídos a partir do estágio CHAT e do `DenseEmbedSettings`.

Variáveis novas (todas opcionais, com defaults razoáveis):
    DENSE_EMBED_PROVIDER, DENSE_EMBED_MODEL, DENSE_EMBED_DIM
    SPARSE_EMBED_MODEL
    CHAT_LLM_PROVIDER, CHAT_LLM_MODEL, CHAT_LLM_TEMPERATURE
    DOC_SUMMARY_LLM_PROVIDER, DOC_SUMMARY_LLM_MODEL, DOC_SUMMARY_LLM_TEMPERATURE
    CHUNK_METADATA_LLM_PROVIDER, CHUNK_METADATA_LLM_MODEL, CHUNK_METADATA_LLM_TEMPERATURE
    RERANK_BACKEND, RERANK_CROSS_ENCODER_MODEL, RERANK_LLM_PROVIDER, RERANK_LLM_MODEL
    INGEST_MAX_WORKERS
    RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP
    SEMANTIC_BREAKPOINT_THRESHOLD_TYPE, SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT
    HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_CHILD_OVERLAP
    STAGE1_TOP_K, STAGE2_TOP_K_PER_STRATEGY, RRF_K, FINAL_TOP_K

Compatibilidade legada:
    MISTRAL_MODEL → usado como fallback para CHAT_LLM_MODEL se não for setado.
    GOOGLE_MODEL  → usado como fallback para DENSE_EMBED_MODEL se não for setado.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from server.src.fii_rag.llm.stages import LLMStage

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


_COMMON_SETTINGS = dict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class ApiKeys(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS)

    mistral_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS)

    qdrant_url: str = "http://localhost:6333"


class DenseEmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS, env_prefix="DENSE_EMBED_")

    provider: str = "google"
    model: str = "text-embedding-004"
    dim: int = 768


class SparseEmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS, env_prefix="SPARSE_EMBED_")

    model: str = "Qdrant/bm25"


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS)

    recursive_chunk_size: int = 1000
    recursive_chunk_overlap: int = 200
    semantic_breakpoint_threshold_type: str = "percentile"
    semantic_breakpoint_threshold_amount: float = 95.0
    hierarchical_parent_size: int = 2000
    hierarchical_child_size: int = 400
    hierarchical_child_overlap: int = 50


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS)

    stage1_top_k: int = 8
    stage2_top_k_per_strategy: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    rerank_backend: str = "cross_encoder"
    rerank_cross_encoder_model: str = "BAAI/bge-reranker-base"


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS)

    ingest_max_workers: int = 4


@dataclass
class LLMStageSettings:
    """Settings de um estágio LLM. Não é um BaseSettings porque os prefixos de
    env são dinâmicos por estágio — preenchido em `AppConfig._load_llm_stages`.
    """

    provider: str
    model: str
    temperature: float


_LLM_STAGE_DEFAULTS: dict[LLMStage, tuple[str, str, str, float]] = {
    LLMStage.CHAT: ("CHAT_LLM", "mistral", "mistral-large-latest", 0.1),
    LLMStage.DOC_SUMMARY: ("DOC_SUMMARY_LLM", "mistral", "mistral-large-latest", 0.0),
    LLMStage.CHUNK_METADATA: (
        "CHUNK_METADATA_LLM",
        "mistral",
        "mistral-small-latest",
        0.0,
    ),
    LLMStage.RERANK: ("RERANK_LLM", "mistral", "mistral-small-latest", 0.0),
}


class AppConfig:
    """Configuração agregada do app.

    Carrega `.env` (via `python-dotenv`) e instancia as seções via
    `pydantic-settings`. Os estágios de LLM são lidos imperativamente porque
    o prefixo varia por estágio.
    """

    def __init__(self) -> None:
        load_dotenv()

        self.api_keys = ApiKeys()
        self.qdrant = QdrantSettings()
        self.dense_embed = self._load_dense_embed()
        self.sparse_embed = SparseEmbedSettings()
        self.chunking = ChunkingSettings()
        self.retrieval = RetrievalSettings()
        self.ingestion = IngestionSettings()
        self.llm_stages = self._load_llm_stages()

    @staticmethod
    def _load_dense_embed() -> DenseEmbedSettings:
        """Carrega `DenseEmbedSettings` com fallback do legado `GOOGLE_MODEL`.

        Se `DENSE_EMBED_MODEL` não estiver setado mas `GOOGLE_MODEL` estiver
        (e o provider resolvido for `google`), usa o legado.
        """
        settings = DenseEmbedSettings()
        if (
            os.getenv("DENSE_EMBED_MODEL") is None
            and settings.provider.lower() == "google"
            and (legacy := os.getenv("GOOGLE_MODEL"))
        ):
            settings = settings.model_copy(update={"model": legacy})
        return settings

    @staticmethod
    def _load_llm_stages() -> dict[LLMStage, LLMStageSettings]:
        """Lê `<PREFIX>_PROVIDER`, `<PREFIX>_MODEL`, `<PREFIX>_TEMPERATURE` por estágio.

        Compatibilidade legada: se `CHAT_LLM_MODEL` não estiver setado mas
        `MISTRAL_MODEL` estiver, usa o legado para o estágio CHAT.
        """
        legacy_chat_model = os.getenv("MISTRAL_MODEL")
        stages: dict[LLMStage, LLMStageSettings] = {}
        for stage, (
            prefix,
            default_provider,
            default_model,
            default_temp,
        ) in _LLM_STAGE_DEFAULTS.items():
            provider = os.getenv(f"{prefix}_PROVIDER", default_provider)
            model = os.getenv(f"{prefix}_MODEL")
            if model is None:
                model = (
                    legacy_chat_model
                    if (stage == LLMStage.CHAT and legacy_chat_model)
                    else default_model
                )
            temperature_str = os.getenv(f"{prefix}_TEMPERATURE")
            temperature = (
                float(temperature_str) if temperature_str is not None else default_temp
            )
            stages[stage] = LLMStageSettings(
                provider=provider, model=model, temperature=temperature
            )
        return stages

    # ------------------------------------------------------------------
    # Back-compat: superfície usada por main.py e frontend/app.py legados.
    # ------------------------------------------------------------------

    @property
    def qdrant_url(self) -> str:
        return self.qdrant.qdrant_url

    def get_llm_and_embeddings(self) -> tuple[Optional["BaseChatModel"], Optional["Embeddings"]]:
        """Constrói `(chat_llm, dense_embeddings_langchain)` a partir do estágio CHAT
        e do `DenseEmbedSettings`. Mantém o contrato do código legado.
        """
        from server.src.fii_rag.embeddings.dense import DenseEmbedderFactory
        from server.src.fii_rag.llm.factory import LLMFactory

        if not self.api_keys.mistral_api_key:
            print("Aviso: MISTRAL_API_KEY não encontrada no .env.")
        if not self.api_keys.google_api_key:
            print("Aviso: GOOGLE_API_KEY não encontrada no .env.")

        llm: Optional["BaseChatModel"] = None
        embed_model: Optional["Embeddings"] = None

        try:
            llm = LLMFactory(self).for_stage(LLMStage.CHAT)
        except Exception as e:  # noqa: BLE001
            print(f"Aviso: não foi possível instanciar o CHAT LLM ({e}).")

        try:
            embed_model = DenseEmbedderFactory(self).build_langchain()
        except Exception as e:  # noqa: BLE001
            print(f"Aviso: não foi possível instanciar o dense embedder ({e}).")

        return llm, embed_model
