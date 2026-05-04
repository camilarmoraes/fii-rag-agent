from typing import Optional

from pydantic import BaseModel, Field


class NumericFacts(BaseModel):
    """Numéricos-chave extraídos por chunk de relatório de FII.

    Achatados como `num_*` no `ChunkPayload` para permitir filtros range nativos
    do Qdrant (`FieldCondition` + `Range`).
    """

    dividend_yield_pct: Optional[float] = Field(
        None, description="Dividend Yield em percentual (mensal ou anual)."
    )
    p_vp: Optional[float] = Field(None, description="Razão Preço / Valor Patrimonial.")
    vacancia_pct: Optional[float] = Field(None, description="Taxa de vacância em %.")
    patrimonio_liquido_brl: Optional[float] = Field(
        None, description="Patrimônio líquido em reais."
    )
    cota_brl: Optional[float] = Field(None, description="Valor da cota em reais.")
    n_imoveis: Optional[int] = Field(None, description="Número de imóveis no portfólio.")
