"""Extração de `NumericFacts` por chunk: regex-first com fallback LLM heurístico.

Patterns:
    - DY:        `DY` ou `dividend yield` seguido de número e %
    - P/VP:      `P/VP` ou `P VP` seguido de número
    - Vacância:  `vacância` seguida de número e %
    - Patrimônio: `patrimônio líquido` + R$ + valor + (milhões|bilhões)?
    - Cota:      `cota` ou `valor da cota` seguido de R$ + valor

LLM fallback: quando regex retornou tudo None, chunk < 800 chars, e contém ≥1
termo-chave (`DY`, `R$`, `%`, `patrimônio`, `vacância`), invoca um LLM
pequeno com `with_structured_output(NumericFacts)`. Se LLM falhar, devolve
`NumericFacts()` vazio.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from server.src.fii_rag.schemas.numerics import NumericFacts

DY_RE = re.compile(
    r"(?:dividend\s*yield|\bDY\b)[^\d%]{0,30}([\d]{1,3}[.,]?\d{0,2})\s*%",
    re.IGNORECASE,
)
PVP_RE = re.compile(
    r"\bP\s*[\/\\]?\s*VP\b[^\d]{0,15}([\d]{1,3}[.,]?\d{0,2})",
    re.IGNORECASE,
)
VAC_RE = re.compile(
    r"vac[âa]ncia[^\d%]{0,30}([\d]{1,3}[.,]?\d{0,2})\s*%",
    re.IGNORECASE,
)
PL_RE = re.compile(
    r"patrim[ôo]nio\s*l[íi]quido[^\d]{0,15}R?\$?\s*([\d.]+(?:[.,]\d+)?)\s*"
    r"(milh[õo]es|bilh[õo]es|mi|bi)?",
    re.IGNORECASE,
)
COTA_RE = re.compile(
    r"(?:valor\s*da\s*cota|\bcota\b)[^\d]{0,15}R?\$?\s*([\d]{1,4}[.,]\d{2})",
    re.IGNORECASE,
)
N_IMOVEIS_RE = re.compile(
    r"(\d{1,3})\s*(?:im[óo]veis|empreendimentos|ativos\s*imobili[áa]rios)",
    re.IGNORECASE,
)

LLM_FALLBACK_TRIGGER_TERMS = (
    "DY", "dividend", "R$", "%", "patrimônio", "patrimonio",
    "vacância", "vacancia", "P/VP", "cota",
)
LLM_FALLBACK_MAX_CHARS = 800


def _to_float(s: Optional[str]) -> Optional[float]:
    """Converte string numérica pt-BR ou en-US para float.

    Regra: se há vírgula, é o separador decimal (`.` vira separador de milhar);
    senão, o `.` já é o decimal (formato en-US). Cobre `0,85`, `1.05`, `4.200,50`.
    """
    if s is None:
        return None
    s = s.strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pl(text: str) -> Optional[float]:
    m = PL_RE.search(text)
    if not m:
        return None
    base = _to_float(m.group(1))
    if base is None:
        return None
    unit = (m.group(2) or "").lower()
    if "bi" in unit:
        return base * 1_000_000_000
    if "mi" in unit:
        return base * 1_000_000
    return base


class NumericFactsExtractor:
    def __init__(self, llm: Any = None, use_llm_fallback: bool = True):
        """`llm` deve ser um `BaseChatModel`; pode ser None se `use_llm_fallback=False`."""
        self.use_llm_fallback = use_llm_fallback and llm is not None
        self._llm_struct = llm.with_structured_output(NumericFacts) if self.use_llm_fallback else None

    def extract(self, text: str) -> NumericFacts:
        regex_facts = self._extract_regex(text)
        if not self._is_empty(regex_facts):
            return regex_facts
        if self.use_llm_fallback and self._should_call_llm(text):
            try:
                llm_facts = self._llm_struct.invoke(
                    "Extraia métricas numéricas (DY %, P/VP, vacância %, "
                    "patrimônio em R$, cota em R$, número de imóveis) do trecho "
                    "abaixo. Use None para os campos ausentes.\n\n"
                    f"--- TRECHO ---\n{text}"
                )
                return llm_facts
            except Exception:
                pass
        return regex_facts

    @staticmethod
    def _extract_regex(text: str) -> NumericFacts:
        n_im_match = N_IMOVEIS_RE.search(text)
        n_im = int(n_im_match.group(1)) if n_im_match else None
        return NumericFacts(
            dividend_yield_pct=_to_float(DY_RE.search(text).group(1)) if DY_RE.search(text) else None,
            p_vp=_to_float(PVP_RE.search(text).group(1)) if PVP_RE.search(text) else None,
            vacancia_pct=_to_float(VAC_RE.search(text).group(1)) if VAC_RE.search(text) else None,
            patrimonio_liquido_brl=_parse_pl(text),
            cota_brl=_to_float(COTA_RE.search(text).group(1)) if COTA_RE.search(text) else None,
            n_imoveis=n_im,
        )

    @staticmethod
    def _is_empty(f: NumericFacts) -> bool:
        return all(
            getattr(f, k) is None
            for k in (
                "dividend_yield_pct",
                "p_vp",
                "vacancia_pct",
                "patrimonio_liquido_brl",
                "cota_brl",
                "n_imoveis",
            )
        )

    @staticmethod
    def _should_call_llm(text: str) -> bool:
        if len(text) > LLM_FALLBACK_MAX_CHARS:
            return False
        return any(term.lower() in text.lower() for term in LLM_FALLBACK_TRIGGER_TERMS)
