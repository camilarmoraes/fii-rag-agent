from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

ReportType = Literal["gerencial", "demonstracao", "fato_relevante", "outro"]


class DocumentMetadataExtracted(BaseModel):
    """Campos de nível-doc extraídos pelo `DOC_SUMMARY_LLM_MODEL`.

    Não inclui `doc_id`, `source_filename`, `ingested_at`, `total_pages` —
    esses são preenchidos pelo `IngestionPipeline` após a chamada do LLM.
    """

    ticker: Optional[str] = Field(
        None,
        description=(
            "Código do FII na bolsa (4 letras maiúsculas + '11', ex: HGLG11). "
            "None se não identificado."
        ),
    )
    cnpj: Optional[str] = Field(
        None,
        description=(
            "CNPJ do fundo apenas com 14 dígitos, sem pontuação. "
            "None se não identificado."
        ),
    )
    fund_name: Optional[str] = Field(None, description="Nome completo do fundo.")
    report_date: Optional[date] = Field(
        None, description="Data de referência do relatório (ISO 8601)."
    )
    report_month: Optional[int] = Field(
        None, ge=1, le=12, description="Mês de referência (1-12)."
    )
    report_year: Optional[int] = Field(None, description="Ano de referência (4 dígitos).")
    report_quarter: Optional[int] = Field(
        None, ge=1, le=4, description="Trimestre de referência (1-4)."
    )
    report_type: Optional[ReportType] = Field(
        None,
        description=(
            "Tipo do relatório: gerencial, demonstracao, fato_relevante, ou outro."
        ),
    )
    summary: str = Field(
        "",
        description=(
            "Resumo executivo do relatório, focando em performance, ativos, "
            "riscos e perspectivas."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="5 a 10 palavras-chave fundamentais para busca.",
    )


class DocumentMetadata(DocumentMetadataExtracted):
    """Metadado completo de um documento, com campos gerados pelo sistema."""

    doc_id: str
    source_filename: str
    ingested_at: datetime
    total_pages: int
