from enum import Enum


class LLMStage(str, Enum):
    """Estágios distintos do pipeline onde um LLM é usado.

    Cada estágio é configurado independentemente via `.env` com prefixo:
    - `CHAT_LLM_*`            : resposta final do agente
    - `DOC_SUMMARY_LLM_*`     : sumário do documento + extração de metadados-doc
    - `CHUNK_METADATA_LLM_*`  : metadados por chunk (title/keywords/summary)
    - `RERANK_LLM_*`          : LLM-as-reranker (alternativa ao cross-encoder)

    Sufixos por estágio: `_PROVIDER`, `_MODEL`, `_TEMPERATURE`.
    """

    CHAT = "chat"
    DOC_SUMMARY = "doc_summary"
    CHUNK_METADATA = "chunk_metadata"
    RERANK = "rerank"
