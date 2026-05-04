"""Extração de metadados-doc via `DOC_SUMMARY_LLM_MODEL` + fallback regex."""

from __future__ import annotations

import re
from typing import Any, Optional

from server.src.fii_rag.schemas.document import DocumentMetadataExtracted

TICKER_RE = re.compile(r"\b([A-Z]{4}11)\b")
CNPJ_RE = re.compile(r"(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})")

TRUNCATE_HEAD = 15000
TRUNCATE_TAIL = 5000


def fallback_ticker(text: str) -> Optional[str]:
    m = TICKER_RE.search(text)
    return m.group(1) if m else None


def fallback_cnpj(text: str) -> Optional[str]:
    m = CNPJ_RE.search(text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    return digits[:14] if len(digits) >= 14 else None


def truncate(text: str, head: int = TRUNCATE_HEAD, tail: int = TRUNCATE_TAIL) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n\n[...]\n\n" + text[-tail:]


PROMPT = (
    "Você é um analista sênior de Fundos de Investimento Imobiliário (FIIs) "
    "brasileiros. Analise o relatório abaixo e extraia metadados estruturados.\n\n"
    "Identifique:\n"
    "- ticker (código na bolsa, formato XXXX11)\n"
    "- cnpj (apenas 14 dígitos, sem pontuação)\n"
    "- fund_name (nome completo do fundo)\n"
    "- report_date / month / year / quarter (período de referência do relatório)\n"
    "- report_type (gerencial | demonstracao | fato_relevante | outro)\n"
    "- summary (resumo executivo focando em performance, ativos, riscos, perspectivas)\n"
    "- keywords (5 a 10 palavras-chave fundamentais para busca)\n\n"
    "Use None / lista vazia para campos não identificáveis. Não invente.\n\n"
    "--- RELATÓRIO ---\n"
)


class DocMetadataExtractor:
    def __init__(self, llm: Any):
        self.llm = llm.with_structured_output(DocumentMetadataExtracted)

    def extract(self, full_text: str) -> DocumentMetadataExtracted:
        text = truncate(full_text)
        try:
            extracted = self.llm.invoke(PROMPT + text)
        except Exception as e:  # noqa: BLE001
            print(f"[DocMetadataExtractor] LLM falhou ({e}). Usando schema vazio.")
            extracted = DocumentMetadataExtracted()

        # Fallbacks regex no início do texto (cabeçalho normalmente tem identidade)
        head = full_text[:5000]
        if not extracted.ticker:
            extracted.ticker = fallback_ticker(head)
        if not extracted.cnpj:
            extracted.cnpj = fallback_cnpj(head)
        return extracted
