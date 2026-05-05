"""Tela de gerenciamento de coleções: criar lógica/legacy + listar + deletar."""

from __future__ import annotations

import streamlit as st
from qdrant_client.models import Distance

from frontend.components.di import get_provisioner, get_raw_client
from frontend.components.helpers import classify_collections
from frontend.components.ui import render_hero, render_metric_box
from server.src.fii_rag.chunking import ALL_STRATEGIES
from server.src.fii_rag.store import (
    CollectionNaming,
    QdrantRepository,
    exclude_config_filter,
)

DIST_MAP = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
}


def render_collections() -> None:
    render_hero(
        "Gerenciar Collections",
        "Visualize, crie e explore as collections do Qdrant.",
    )

    try:
        client = get_raw_client()
        logicals, legacies = classify_collections(client)
    except Exception as e:  # noqa: BLE001
        st.error(f"Não foi possível conectar ao Qdrant: {e}")
        st.stop()

    _render_create_logical_form(client)
    _render_create_legacy_form(client)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    _render_listing(client, logicals, legacies)


# ────────────────────────────────────────────────────────────
# Forms de criação
# ────────────────────────────────────────────────────────────

def _render_create_logical_form(client) -> None:
    non_summary = [s for s in ALL_STRATEGIES if s != "summary"]
    with st.expander("➕ Criar nova coleção lógica (multi-strategy)", expanded=False):
        st.caption(
            "💡 Uma coleção lógica provisiona internamente N sub-coleções "
            "(`<nome>__summary`, `<nome>__recursive`, etc.) — cada uma indexa "
            "o documento com uma estratégia de chunking distinta. Você só vê o "
            "nome lógico aqui."
        )
        with st.form("create_logical_form"):
            new_name = st.text_input(
                "Nome da coleção", placeholder="ex: fii_2026 (apenas letras/dígitos/_)"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                vector_size = st.selectbox(
                    "Dimensão do vetor denso",
                    options=[768, 1024, 1536, 3072],
                    index=3,
                )
                hybrid = st.toggle(
                    "Busca híbrida (BM25 + dense)",
                    value=True,
                    help="Cria também o vetor sparse para fusão RRF nativa.",
                )
            with col_b:
                distance = st.selectbox(
                    "Métrica de distância", list(DIST_MAP.keys())
                )
                selected_extras = st.multiselect(
                    "Estratégias de chunking",
                    options=non_summary,
                    default=non_summary,
                    help="`summary` é sempre incluída automaticamente.",
                )
            submit = st.form_submit_button(
                "Criar coleção lógica", use_container_width=True
            )

        if submit:
            if not new_name.strip():
                st.error("Informe um nome para a coleção.")
            elif not CollectionNaming.is_valid_logical(new_name.strip()):
                st.error(
                    "Nome inválido. Use apenas letras, dígitos e `_`, máx 40 chars."
                )
            else:
                strategies = ["summary"] + list(selected_extras)
                try:
                    result = get_provisioner(client).provision(
                        logical=new_name.strip(),
                        strategies=strategies,
                        hybrid=hybrid,
                        dense_dim=vector_size,
                        distance=DIST_MAP[distance],
                    )
                    st.success(
                        f"✅ Coleção lógica **{new_name}** criada com "
                        f"{len(result['physical_names'])} físicas: "
                        f"{', '.join(result['physical_names'])}"
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Erro ao provisionar: {e}")


def _render_create_legacy_form(client) -> None:
    with st.expander("➕ Criar coleção avulsa (legacy single-coll)", expanded=False):
        st.caption(
            "Modo simples: 1 vetor por coleção, sem multi-strategy. "
            "Mantido para compatibilidade — recomendamos coleções lógicas."
        )
        with st.form("create_legacy_form"):
            legacy_name = st.text_input("Nome", placeholder="ex: fii_legacy")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                size = st.selectbox(
                    "Dimensão", options=[768, 1024, 1536, 3072], index=3,
                    key="legacy_size",
                )
            with col_b:
                dist = st.selectbox(
                    "Métrica", list(DIST_MAP.keys()), key="legacy_distance"
                )
            with col_c:
                hybrid = st.toggle("Hybrid (BM25)", value=True, key="legacy_hybrid")
            submit = st.form_submit_button("Criar (legacy)", use_container_width=True)

        if submit:
            if not legacy_name.strip():
                st.error("Informe um nome.")
            else:
                try:
                    QdrantRepository(client).ensure_collection(
                        name=legacy_name.strip(),
                        dim=size,
                        distance=DIST_MAP[dist],
                        hybrid=hybrid,
                    )
                    st.success(f"✅ Coleção legacy **{legacy_name}** criada.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Erro ao criar legacy: {e}")


# ────────────────────────────────────────────────────────────
# Listagem
# ────────────────────────────────────────────────────────────

def _render_listing(client, logicals: list[dict], legacies: list[str]) -> None:
    total = len(logicals) + len(legacies)
    if total == 0:
        st.info("Nenhuma coleção encontrada. Crie uma acima.")
        return

    _render_global_metrics(client, logicals, legacies)

    st.markdown("<br>", unsafe_allow_html=True)

    if logicals:
        st.markdown("### 🧩 Coleções lógicas")
        for li in logicals:
            _render_logical_card(client, li)

    if legacies:
        st.markdown("### 🗄️ Coleções legadas")
        for name in legacies:
            _render_legacy_card(client, name)


def _render_global_metrics(client, logicals: list[dict], legacies: list[str]) -> None:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(render_metric_box(len(logicals), "Lógicas"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_box(len(legacies), "Legadas"), unsafe_allow_html=True)
    with m3:
        total_pts = 0
        for li in logicals:
            for s in li["strategies"]:
                try:
                    total_pts += (
                        client.get_collection(
                            CollectionNaming.to_physical(li["logical"], s)
                        ).points_count
                        or 0
                    )
                except Exception:  # noqa: BLE001
                    pass
        for n in legacies:
            try:
                total_pts += client.get_collection(n).points_count or 0
            except Exception:  # noqa: BLE001
                pass
        st.markdown(render_metric_box(total_pts, "Pontos totais"), unsafe_allow_html=True)


def _render_logical_card(client, li: dict) -> None:
    logical = li["logical"]
    cfg = li["config"]
    hybrid = cfg.get("hybrid", "—")
    dim = cfg.get("dense_dim", "—")
    distance = str(cfg.get("distance", "—")).replace("Distance.", "")
    chips = " · ".join(li["strategies"])

    with st.expander(f"🧩 **{logical}** · {chips}", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estratégias", len(li["strategies"]))
        c2.metric("Dimensão", dim)
        c3.metric("Distância", distance)
        c4.metric("Hybrid", "Sim" if hybrid else "Não")

        summary_coll = CollectionNaming.to_physical(logical, "summary")
        try:
            n_docs = client.count(
                collection_name=summary_coll,
                count_filter=exclude_config_filter(),
                exact=True,
            ).count
        except Exception:  # noqa: BLE001
            n_docs = "—"

        n_chunks = 0
        for s in li["strategies"]:
            if s == "summary":
                continue
            try:
                n_chunks += (
                    client.get_collection(
                        CollectionNaming.to_physical(logical, s)
                    ).points_count
                    or 0
                )
            except Exception:  # noqa: BLE001
                pass

        st.markdown(
            f"📄 Documentos: **{n_docs}** · ✂️ Chunks (somados): **{n_chunks}**"
        )

        with st.form(f"delete_logical_{logical}"):
            confirm = st.checkbox(
                f"Confirmo que quero deletar a lógica **{logical}** "
                f"(apaga {len(li['strategies'])} físicas em cascata)"
            )
            if st.form_submit_button("🗑️ Deletar lógica", type="secondary"):
                if confirm:
                    try:
                        deleted = get_provisioner(client).delete_logical(logical)
                        st.success(
                            f"Lógica **{logical}** deletada ({deleted} físicas apagadas)."
                        )
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Erro ao deletar: {e}")
                else:
                    st.warning("Marque a confirmação para deletar.")


def _render_legacy_card(client, name: str) -> None:
    with st.expander(f"🗄️ **{name}** `[Legacy]`", expanded=False):
        try:
            info = client.get_collection(name)
            pts = info.points_count or 0
            vec_size = None
            dist_name = "—"
            cfg = info.config.params.vectors
            if isinstance(cfg, dict):
                first = next(iter(cfg.values()), None)
                if first:
                    vec_size = first.size
                    dist_name = str(first.distance).replace("Distance.", "")
            else:
                vec_size = getattr(cfg, "size", None)
                dist_name = str(getattr(cfg, "distance", "—")).replace("Distance.", "")

            sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
            is_hybrid = bool(sparse_cfg)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documentos", pts)
            c2.metric("Dimensão", vec_size or "—")
            c3.metric("Distância", dist_name)
            c4.metric("Hybrid", "Sim" if is_hybrid else "Não")

            if pts > 0:
                st.markdown("**Amostra:**")
                points = client.scroll(
                    collection_name=name,
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )[0]
                for p in points:
                    meta = p.payload or {}
                    content = meta.get("page_content", meta.get("text", "—"))
                    title = meta.get("metadata", {}).get(
                        "title", meta.get("title", "Sem título")
                    )
                    keywords = meta.get("metadata", {}).get(
                        "keywords", meta.get("keywords", [])
                    )
                    st.markdown(f"**{title}**")
                    if keywords:
                        kws = (
                            " · ".join(keywords[:5])
                            if isinstance(keywords, list)
                            else str(keywords)
                        )
                        st.caption(f"🏷️ {kws}")
                    truncated = content[:400] + ("..." if len(content) > 400 else "")
                    st.markdown(
                        f'<div class="info-card" style="font-size:0.85rem;color:#94a3b8;">'
                        f"{truncated}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro ao obter detalhes: {e}")

        with st.form(f"delete_legacy_{name}"):
            confirm = st.checkbox(f"Confirmo deletar legacy **{name}**")
            if st.form_submit_button("🗑️ Deletar legacy", type="secondary"):
                if confirm:
                    try:
                        client.delete_collection(name)
                        st.success(f"Legacy {name} deletada.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Erro ao deletar: {e}")
                else:
                    st.warning("Marque a confirmação para deletar.")
