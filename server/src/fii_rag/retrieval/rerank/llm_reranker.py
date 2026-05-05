"""Reranker baseado em LLM — alternativa ao cross-encoder local.

Útil quando não se quer baixar `sentence-transformers` ou rodar GPU. Gasta
crédito de API por query (1 chamada para reordenar até `rerank_top_n`).
Selecionado via `RERANK_BACKEND=llm` no `.env`.
"""

from __future__ import annotations

from typing import Any

PROMPT = (
    "Você é um especialista em ranking de relevância para Fundos Imobiliários. "
    "Reordene os trechos abaixo da maior para a menor relevância em relação à "
    "pergunta do usuário. Responda APENAS com a lista de índices separados por "
    "vírgula, do mais relevante para o menos relevante. Não inclua explicações.\n\n"
    "Pergunta: {query}\n\n"
    "Trechos:\n{passages}\n\n"
    "Índices (apenas números separados por vírgula):"
)
MAX_PASSAGE_CHARS = 600


class LLMReranker:
    def __init__(self, llm: Any):
        self.llm = llm

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        numbered = "\n".join(
            f"[{i}] {p[:MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages)
        )
        try:
            response = self.llm.invoke(PROMPT.format(query=query, passages=numbered))
            content = getattr(response, "content", str(response))
            # Parse "0, 3, 1, 2" → atribui scores decrescentes na ordem
            indices: list[int] = []
            for tok in content.replace(",", " ").split():
                if tok.strip().isdigit():
                    idx = int(tok.strip())
                    if 0 <= idx < len(passages) and idx not in indices:
                        indices.append(idx)
            scores = [0.0] * len(passages)
            n = max(len(indices), 1)
            for rank, idx in enumerate(indices):
                scores[idx] = 1.0 - rank / n
            return scores
        except Exception as e:  # noqa: BLE001
            print(f"[LLMReranker] falhou ({e}); devolvendo scores neutros.")
            return [1.0] * len(passages)
