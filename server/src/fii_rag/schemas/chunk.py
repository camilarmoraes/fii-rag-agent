from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from server.src.fii_rag.schemas.numerics import NumericFacts

ChunkingStrategyName = Literal[
    "summary", "recursive", "semantic", "doc_aware", "hierarchical"
]


class ChunkExtractedMetadata(BaseModel):
    """Campos por-chunk extraídos pelo `CHUNK_METADATA_LLM_MODEL`."""

    title: str = Field(description="Título conciso do trecho (5-10 palavras).")
    keywords: list[str] = Field(
        description="5 ou mais palavras-chave fundamentais para busca."
    )
    chunk_summary: str = Field(description="Resumo curto do trecho (1-2 frases).")


class ChunkMetadata(BaseModel):
    """Metadado completo de um chunk: herda doc-meta + campos locais + numéricos."""

    doc_id: str
    source_filename: str
    ticker: Optional[str] = None
    cnpj: Optional[str] = None
    fund_name: Optional[str] = None
    report_date: Optional[date] = None
    report_month: Optional[int] = None
    report_year: Optional[int] = None
    report_quarter: Optional[int] = None
    report_type: Optional[str] = None

    chunk_id: str
    strategy: ChunkingStrategyName
    chunk_index: int
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None

    title: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_summary: str = ""

    numerics: NumericFacts = Field(default_factory=NumericFacts)


class ChunkPayload(BaseModel):
    """Payload achatado para o ponto Qdrant.

    Numéricos viram campos `num_*` para `FieldCondition` + `Range` direto, sem
    aninhamento. Datas viram strings ISO 8601 (Qdrant aceita string em
    `MatchValue`/`Range` lexicográfico, mas para filtros temporais reais usar
    `report_year` + `report_month` + `report_quarter` é mais confiável).
    """

    text: str

    doc_id: str
    source_filename: str
    ticker: Optional[str] = None
    cnpj: Optional[str] = None
    fund_name: Optional[str] = None
    report_date: Optional[str] = None
    report_month: Optional[int] = None
    report_year: Optional[int] = None
    report_quarter: Optional[int] = None
    report_type: Optional[str] = None

    chunk_id: str
    strategy: str
    chunk_index: int
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None

    title: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_summary: str = ""

    num_dy: Optional[float] = None
    num_pvp: Optional[float] = None
    num_vacancia: Optional[float] = None
    num_patrimonio: Optional[float] = None
    num_cota: Optional[float] = None
    num_n_imoveis: Optional[int] = None

    @classmethod
    def from_chunk_metadata(cls, meta: ChunkMetadata, text: str) -> "ChunkPayload":
        n = meta.numerics
        return cls(
            text=text,
            doc_id=meta.doc_id,
            source_filename=meta.source_filename,
            ticker=meta.ticker,
            cnpj=meta.cnpj,
            fund_name=meta.fund_name,
            report_date=meta.report_date.isoformat() if meta.report_date else None,
            report_month=meta.report_month,
            report_year=meta.report_year,
            report_quarter=meta.report_quarter,
            report_type=meta.report_type,
            chunk_id=meta.chunk_id,
            strategy=meta.strategy,
            chunk_index=meta.chunk_index,
            page_number=meta.page_number,
            section_heading=meta.section_heading,
            parent_chunk_id=meta.parent_chunk_id,
            parent_text=meta.parent_text,
            title=meta.title,
            keywords=meta.keywords,
            chunk_summary=meta.chunk_summary,
            num_dy=n.dividend_yield_pct,
            num_pvp=n.p_vp,
            num_vacancia=n.vacancia_pct,
            num_patrimonio=n.patrimonio_liquido_brl,
            num_cota=n.cota_brl,
            num_n_imoveis=n.n_imoveis,
        )
