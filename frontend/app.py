import sys
import os

# Garantir que o módulo raiz do projeto está no path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from server.src.fii_rag.config import AppConfig
from server.src.fii_rag.db import QdrantStoreProvider
from server.src.fii_rag.chunking import LangChainParser, LangChainSemanticExtractor
from server.src.fii_rag.ingestion import PDFIngestionManager
from server.src.fii_rag.retriever import HybridQueryEngineBuilder
from server.src.fii_rag.agent import RAGAgent

# ──────────────────────────────────────────────
# Configuração da Página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FII RAG Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Customizado
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fundo escuro premium */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* Cards / containers */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Título principal */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Badge de status */
.badge-ok {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-err {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(79,70,229,0.3);
}
.chat-bot {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.95rem;
}
.chat-label {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 0.2rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Separador */
.section-divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.15);
    margin: 1.5rem 0;
}

/* Métrica */
.metric-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cache: Inicialização dos componentes (DI)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Inicializando componentes...")
def get_components():
    """Inicializa e cacheia todos os componentes do sistema."""
    config = AppConfig()
    llm, embed_model = config.get_llm_and_embeddings()
    qdrant_provider = QdrantStoreProvider(url=config.qdrant_url, embed_model=embed_model)
    parser = LangChainParser()
    extractor = LangChainSemanticExtractor(llm=llm)
    ingestion_manager = PDFIngestionManager(
        vector_store_provider=qdrant_provider,
        document_parser=parser,
        metadata_extractor=extractor,
    )
    query_builder = HybridQueryEngineBuilder(
        vector_store_provider=qdrant_provider,
        top_k=10,
        rerank_top_k=3,
    )
    agent = RAGAgent(query_engine_builder=query_builder, llm=llm)
    raw_client = QdrantClient(url=config.qdrant_url)
    return config, qdrant_provider, ingestion_manager, agent, raw_client

@st.cache_resource(show_spinner=False)
def check_qdrant_connection():
    try:
        config = AppConfig()
        client = QdrantClient(url=config.qdrant_url)
        client.get_collections()
        return True, config.qdrant_url
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 FII RAG Agent")
    st.markdown("---")

    qdrant_ok, qdrant_info = check_qdrant_connection()
    if qdrant_ok:
        st.markdown(f'<span class="badge-ok">● Qdrant conectado</span>', unsafe_allow_html=True)
        st.caption(qdrant_info)
    else:
        st.markdown(f'<span class="badge-err">● Qdrant offline</span>', unsafe_allow_html=True)
        st.caption(qdrant_info)

    st.markdown("---")
    page = st.radio(
        "Navegação",
        options=["💬 Chat", "📚 Collections", "📄 Inserir Documento"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<div style="color:#475569;font-size:0.78rem;">Powered by Camila Ribeiro</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_raw_client():
    config = AppConfig()
    return QdrantClient(url=config.qdrant_url)


# ══════════════════════════════════════════════
# PÁGINA: CHAT
# ══════════════════════════════════════════════
if page == "💬 Chat":
    st.markdown('<div class="hero-title">Chat com seus FIIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Faça perguntas sobre os relatórios ingeridos.</div>', unsafe_allow_html=True)

    # Seletor de collection para o chat
    try:
        client = get_raw_client()
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = ["fii_reports"]

    if not collections:
        st.warning("⚠️ Nenhuma collection encontrada. Crie uma na aba **Collections** e insira documentos.")
        st.stop()

    col_chat, col_cfg = st.columns([3, 1])
    with col_cfg:
        selected_collection = st.selectbox("Collection", collections, key="chat_collection")

    # Histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-label">Você</div><div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label">🤖 Agente FII</div><div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Pergunte sobre os relatórios de FIIs...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-label">Você</div><div class="chat-user">{user_input}</div>', unsafe_allow_html=True)

        with st.spinner("⏳ Buscando + Reranking + Mistral AI..."):
            try:
                _, _, _, agent, _ = get_components()
                agent.collection_name = selected_collection
                agent.chain = None  # força rebuild na collection certa
                agent._setup_engine()
                response = agent.chain.invoke({"input": user_input})
                answer = response.get("answer", "Não foi possível gerar resposta.")
            except Exception as e:
                answer = f"❌ Erro ao processar: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f'<div class="chat-label">🤖 Agente FII</div><div class="chat-bot">{answer}</div>', unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("🗑️ Limpar conversa", type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════
# PÁGINA: COLLECTIONS
# ══════════════════════════════════════════════
elif page == "📚 Collections":
    st.markdown('<div class="hero-title">Gerenciar Collections</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Visualize, crie e explore as collections do Qdrant.</div>', unsafe_allow_html=True)

    try:
        client = get_raw_client()
        collections = client.get_collections().collections
    except Exception as e:
        st.error(f"Não foi possível conectar ao Qdrant: {e}")
        st.stop()

    # ── Criar nova collection ──────────────────
    with st.expander("➕ Criar nova collection", expanded=False):
        st.caption(
            "💡 Escolha a dimensão que corresponde ao seu modelo de embeddings. "
            "gemini-embedding-2-preview suporta **768d · 1536d · 3072d**. "
            "A dimensão correta será detectada automaticamente na ingestão."
        )
        with st.form("create_collection_form"):
            new_name = st.text_input("Nome da collection", placeholder="ex: fii_reports_2024")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                vector_size = st.selectbox(
                    "Dimensão do vetor",
                    options=[768, 1024, 1536, 3072],
                    index=3,
                    help="Deve corresponder à dimensão de saída do seu modelo de embeddings.",
                )
            with col_b:
                distance = st.selectbox("Métrica de distância", ["Cosine", "Dot", "Euclid"])
            with col_c:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_create = st.form_submit_button("Criar", use_container_width=True)

        if submit_create:
            if not new_name.strip():
                st.error("Informe um nome para a collection.")
            else:
                dist_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
                try:
                    client.create_collection(
                        collection_name=new_name.strip(),
                        vectors_config=VectorParams(size=vector_size, distance=dist_map[distance]),
                    )
                    st.success(f"✅ Collection **{new_name}** criada com sucesso! ({vector_size}d · {distance})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao criar collection: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Listagem das collections ───────────────
    if not collections:
        st.info("Nenhuma collection encontrada. Crie uma acima.")
    else:
        # Métricas globais
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(collections)}</div><div class="metric-label">Collections</div></div>', unsafe_allow_html=True)
        with m2:
            total_docs = 0
            for c in collections:
                try:
                    info = client.get_collection(c.name)
                    total_docs += info.points_count or 0
                except Exception:
                    pass
            st.markdown(f'<div class="metric-box"><div class="metric-value">{total_docs}</div><div class="metric-label">Documentos totais</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for col in collections:
            with st.expander(f"📂 {col.name}", expanded=False):
                try:
                    info = client.get_collection(col.name)
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

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Documentos", pts)
                    c2.metric("Dimensão", vec_size or "—")
                    c3.metric("Distância", dist_name)

                    # Pré-visualização dos documentos
                    if pts > 0:
                        st.markdown("**Amostra de documentos:**")
                        points = client.scroll(
                            collection_name=col.name,
                            limit=5,
                            with_payload=True,
                            with_vectors=False,
                        )[0]
                        for p in points:
                            meta = p.payload or {}
                            content = meta.get("page_content", meta.get("text", "—"))
                            title = meta.get("metadata", {}).get("title", meta.get("title", "Sem título"))
                            keywords = meta.get("metadata", {}).get("keywords", meta.get("keywords", []))
                            with st.container():
                                st.markdown(f"**{title}**")
                                if keywords:
                                    st.caption(f"🏷️ {' · '.join(keywords[:5]) if isinstance(keywords, list) else keywords}")
                                st.markdown(f'<div class="info-card" style="font-size:0.85rem;color:#94a3b8;">{content[:400]}{"..." if len(content) > 400 else ""}</div>', unsafe_allow_html=True)
                                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erro ao obter detalhes: {e}")

                # Botão de deletar (perigoso — confirmação via checkbox)
                with st.form(f"delete_form_{col.name}"):
                    confirm = st.checkbox(f"Confirmo que quero deletar **{col.name}**")
                    if st.form_submit_button("🗑️ Deletar collection", type="secondary"):
                        if confirm:
                            try:
                                client.delete_collection(col.name)
                                st.success(f"Collection {col.name} deletada.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro ao deletar: {e}")
                        else:
                            st.warning("Marque a confirmação para deletar.")


# ══════════════════════════════════════════════
# PÁGINA: INSERIR DOCUMENTO
# ══════════════════════════════════════════════
elif page == "📄 Inserir Documento":
    st.markdown('<div class="hero-title">Inserir Documento</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Faça upload de um PDF de relatório de FII para ingestão no Qdrant.</div>', unsafe_allow_html=True)

    # Seletor de collection destino
    try:
        client = get_raw_client()
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = []

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if collections:
            target_collection = st.selectbox("Collection destino", collections, key="ingest_collection")
        else:
            st.warning("⚠️ Nenhuma collection encontrada. Crie uma primeiro na aba **📚 Collections**.")
            st.stop()

    # Upload do PDF
    uploaded_file = st.file_uploader(
        "Selecione o arquivo PDF do relatório de FII",
        type=["pdf"],
        help="Apenas arquivos PDF são suportados.",
    )

    if uploaded_file is not None:
        st.markdown(f'<div class="info-card">📄 <b>{uploaded_file.name}</b> · {uploaded_file.size / 1024:.1f} KB</div>', unsafe_allow_html=True)

        # Opções avançadas
        with st.expander("⚙️ Opções avançadas de chunking"):
            chunk_size = st.slider("Tamanho do chunk (tokens)", 200, 2000, 1000, 100)
            chunk_overlap = st.slider("Overlap do chunk (tokens)", 0, 500, 200, 50)
            extract_metadata = st.toggle("Extrair metadados semânticos via Mistral AI", value=True)

        if st.button("🚀 Iniciar Ingestão", type="primary", use_container_width=True):
            # Salva PDF temporariamente
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            progress = st.progress(0, text="Preparando...")
            status = st.empty()

            try:
                config = AppConfig()
                llm, embed_model = config.get_llm_and_embeddings()

                progress.progress(10, text="Conectando ao Qdrant...")
                qdrant_provider = QdrantStoreProvider(url=config.qdrant_url, embed_model=embed_model)
                vector_store = qdrant_provider.get_store(target_collection)

                progress.progress(25, text="Carregando PDF...")
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(tmp_path)
                documents = loader.load()
                status.info(f"📄 {len(documents)} página(s) carregada(s).")

                progress.progress(40, text="Fragmentando em chunks...")
                parser = LangChainParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                splitter = parser.get_parser()
                splits = splitter.split_documents(documents)
                status.info(f"✂️ {len(splits)} chunks gerados.")

                if extract_metadata:
                    progress.progress(55, text=f"Extraindo metadados via Mistral ({len(splits)} chunks)...")
                    extractor = LangChainSemanticExtractor(llm=llm)
                    extractor_fn = extractor.get_extractors()
                    enriched_splits = extractor_fn(splits)
                else:
                    enriched_splits = splits

                progress.progress(80, text="Inserindo no Qdrant...")
                vector_store.add_documents(documents=enriched_splits)

                progress.progress(100, text="Concluído!")
                st.success(f"✅ **{uploaded_file.name}** ingerido com sucesso! {len(enriched_splits)} chunks inseridos na collection **{target_collection}**.")

            except Exception as e:
                st.error(f"❌ Erro durante a ingestão: {e}")
            finally:
                os.unlink(tmp_path)
