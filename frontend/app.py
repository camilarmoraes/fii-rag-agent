"""Entrypoint do frontend Streamlit — sidebar + roteamento.

Estrutura modular (PR 6):
    frontend/
    ├── app.py               # este arquivo: page_config, sidebar, dispatcher
    ├── styles.css           # CSS (carregado via components.ui.inject_styles)
    ├── components/
    │   ├── di.py            # caches @st.cache_resource
    │   ├── helpers.py       # classify_collections, selectbox helpers
    │   └── ui.py            # CSS injection, hero, badges, painel docs
    └── views/               # `pages/` é reservado pelo Streamlit
        ├── chat.py
        ├── collections.py
        └── ingest.py
"""

import os
import sys

# Garantir que a raiz do projeto está no path para `from server.src.fii_rag...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from frontend.components.di import check_qdrant_connection
from frontend.components.ui import inject_styles
from frontend.views.chat import render_chat
from frontend.views.collections import render_collections
from frontend.views.ingest import render_ingest

# ──────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FII RAG Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 FII RAG Agent")
    st.markdown("---")

    qdrant_ok, qdrant_info = check_qdrant_connection()
    if qdrant_ok:
        st.markdown(
            '<span class="badge-ok">● Qdrant conectado</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge-err">● Qdrant offline</span>',
            unsafe_allow_html=True,
        )
    st.caption(qdrant_info)

    st.markdown("---")
    page = st.radio(
        "Navegação",
        options=["💬 Chat", "📚 Collections", "📄 Inserir Documento"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<div style="color:#475569;font-size:0.78rem;">'
        "Powered by Camila Ribeiro</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Roteamento
# ──────────────────────────────────────────────
if page == "💬 Chat":
    render_chat()
elif page == "📚 Collections":
    render_collections()
elif page == "📄 Inserir Documento":
    render_ingest()
