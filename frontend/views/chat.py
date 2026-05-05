"""Tela de chat — selectbox unificado lógicas + legadas, painel de trechos."""

from __future__ import annotations

import streamlit as st

from frontend.components.di import (
    get_components,
    get_raw_client,
    get_retriever_builder,
)
from frontend.components.helpers import (
    build_selector_options,
    classify_collections,
    parse_selector,
)
from frontend.components.ui import (
    render_docs_panel,
    render_hero,
    serialize_docs,
)


def render_chat() -> None:
    render_hero(
        "Chat com seus FIIs",
        "Faça perguntas sobre os relatórios ingeridos.",
    )

    try:
        client = get_raw_client()
        logicals, legacies = classify_collections(client)
    except Exception:  # noqa: BLE001
        logicals, legacies = [], ["fii_reports"]

    options = build_selector_options(logicals, legacies)
    if not options:
        st.warning(
            "⚠️ Nenhuma collection encontrada. Crie uma na aba **Collections** "
            "e insira documentos."
        )
        st.stop()

    _, col_cfg = st.columns([3, 1])
    with col_cfg:
        selected_label = st.selectbox("Collection", options, key="chat_collection")
        kind, selected_name = parse_selector(selected_label)
        if kind == "logical":
            st.caption("🔗 Lógica · two-stage (summary → chunks filtrados + RRF)")
        else:
            st.caption("🗄️ Legacy · busca direta na coll")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Histórico
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-label">Você</div>'
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-label">🤖 Agente FII</div>'
                f'<div class="chat-bot">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("docs"):
                render_docs_panel(msg["docs"])

    user_input = st.chat_input("Pergunte sobre os relatórios de FIIs...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(
            f'<div class="chat-label">Você</div>'
            f'<div class="chat-user">{user_input}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("⏳ Buscando + Reranking + Mistral AI..."):
            docs_used: list = []
            try:
                _, _, _, agent, _ = get_components()
                builder = get_retriever_builder()
                if kind == "logical":
                    agent.set_logical_collection(selected_name, builder)
                else:
                    agent.set_legacy_collection(selected_name, builder)
                response = agent.chain.invoke({"input": user_input})
                answer = response.get("answer", "Não foi possível gerar resposta.")
                docs_used = response.get("context", []) or []
            except Exception as e:  # noqa: BLE001
                answer = f"❌ Erro ao processar: {e}"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "docs": serialize_docs(docs_used),
            }
        )
        st.markdown(
            f'<div class="chat-label">🤖 Agente FII</div>'
            f'<div class="chat-bot">{answer}</div>',
            unsafe_allow_html=True,
        )
        if docs_used:
            render_docs_panel(docs_used)

    if st.session_state.messages:
        if st.button("🗑️ Limpar conversa", type="secondary"):
            st.session_state.messages = []
            st.rerun()
