"""Componentes visuais compartilhados: CSS injection, badges, painel de docs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_STYLES_PATH = Path(__file__).resolve().parent.parent / "styles.css"


def inject_styles() -> None:
    """Injeta o CSS de `frontend/styles.css` na página."""
    css = _STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{subtitle}</div>', unsafe_allow_html=True)


def render_metric_box(value: object, label: str) -> str:
    return (
        f'<div class="metric-box"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>'
    )


def serialize_docs(docs) -> list[dict]:
    """Converte `Document`s do LangChain em dicts leves para session_state."""
    out: list[dict] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        out.append(
            {
                "text": getattr(d, "page_content", str(d)),
                "strategy": meta.get("_strategy", "—"),
                "score": float(meta.get("_score") or 0.0),
                "ticker": meta.get("ticker") or "",
                "year": meta.get("report_year"),
                "quarter": meta.get("report_quarter"),
                "page_number": meta.get("page_number"),
                "section_heading": meta.get("section_heading"),
            }
        )
    return out


def render_docs_panel(docs) -> None:
    """Renderiza o painel 'Trechos usados' com badges. Aceita Documents OU dicts."""
    if not docs:
        return
    items = docs if isinstance(docs[0], dict) else serialize_docs(docs)
    n = len(items)
    with st.expander(f"📚 Trechos usados ({n})", expanded=False):
        for i, item in enumerate(items):
            badges = [f"`#{i + 1}`", f"`{item['strategy']}`"]
            if item.get("ticker"):
                badges.append(f"**{item['ticker']}**")
            year = item.get("year")
            quarter = item.get("quarter")
            if year and quarter:
                badges.append(f"{year}T{quarter}")
            elif year:
                badges.append(str(year))
            if item.get("page_number"):
                badges.append(f"p.{item['page_number']}")
            if item.get("section_heading"):
                badges.append(f"§ {item['section_heading'][:40]}")
            st.markdown(" · ".join(badges))
            text = item.get("text", "")
            preview = text[:350] + ("..." if len(text) > 350 else "")
            st.caption(preview)
            st.markdown("---")
