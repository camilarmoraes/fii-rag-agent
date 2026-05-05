import sys
import os

# Garantir que o módulo raiz do projeto está no path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from server.src.fii_rag.config import AppConfig
from server.src.fii_rag.db import QdrantStoreProvider
from server.src.fii_rag.chunking import (
    ALL_STRATEGIES,
    LangChainParser,
    LangChainSemanticExtractor,
)
from server.src.fii_rag.ingestion import (
    IngestionPipeline,
    PDFIngestionManager,
    build_ingestion_pipeline,
)
from server.src.fii_rag.retriever import HybridQueryEngineBuilder
from server.src.fii_rag.retrieval import RetrieverBuilder, build_retriever_builder
from server.src.fii_rag.agent import RAGAgent
from server.src.fii_rag.store import (
    CollectionNaming,
    LogicalCollectionProvisioner,
    QdrantRepository,
)

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


def get_provisioner(client: QdrantClient) -> LogicalCollectionProvisioner:
    return LogicalCollectionProvisioner(QdrantRepository(client))


@st.cache_resource(show_spinner="⚙️ Inicializando pipeline de ingestão...")
def get_ingestion_pipeline() -> IngestionPipeline:
    return build_ingestion_pipeline(AppConfig())


@st.cache_resource(show_spinner="⚙️ Inicializando retriever builder...")
def get_retriever_builder() -> RetrieverBuilder:
    return build_retriever_builder(AppConfig())


def classify_collections(client: QdrantClient):
    """Separa colls em 2 grupos: lógicas (com info de strategies) e legadas.

    Retorna `(logical_dicts, legacy_names)` onde cada `logical_dict` tem
    `logical`, `strategies`, `config` (lido de `_LOGICAL_CONFIG_` quando
    disponível).
    """
    prov = get_provisioner(client)
    logicals = prov.list_logical()
    legacy = prov.list_legacy()
    logical_dicts = []
    for li in logicals:
        cfg = prov.read_logical_config(li.logical) or {}
        logical_dicts.append(
            {
                "logical": li.logical,
                "strategies": li.strategies,
                "has_summary": li.has_summary,
                "config": cfg,
            }
        )
    return logical_dicts, legacy


# Identificadores no selectbox: prefixo "[L]" para lógicas, "[Legacy]" para soltas
LOGICAL_PREFIX = "[L] "
LEGACY_PREFIX = "[Legacy] "


def _serialize_docs(docs) -> list[dict]:
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


def _render_docs_panel(docs) -> None:
    """Renderiza o painel 'Trechos usados' com badges. `docs` aceita Documents ou dicts."""
    if not docs:
        return
    items = docs if isinstance(docs[0], dict) else _serialize_docs(docs)
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


def build_selector_options(logicals, legacies) -> list[str]:
    return [LOGICAL_PREFIX + l["logical"] for l in logicals] + [
        LEGACY_PREFIX + name for name in legacies
    ]


def parse_selector(label: str) -> tuple[str, str]:
    """Devolve `(kind, name)` onde `kind` ∈ {"logical", "legacy"}."""
    if label.startswith(LOGICAL_PREFIX):
        return "logical", label[len(LOGICAL_PREFIX) :]
    if label.startswith(LEGACY_PREFIX):
        return "legacy", label[len(LEGACY_PREFIX) :]
    return "legacy", label  # fallback


# ══════════════════════════════════════════════
# PÁGINA: CHAT
# ══════════════════════════════════════════════
if page == "💬 Chat":
    st.markdown('<div class="hero-title">Chat com seus FIIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Faça perguntas sobre os relatórios ingeridos.</div>', unsafe_allow_html=True)

    # Seletor de collection para o chat
    try:
        client = get_raw_client()
        logicals, legacies = classify_collections(client)
    except Exception:
        logicals, legacies = [], ["fii_reports"]

    options = build_selector_options(logicals, legacies)
    if not options:
        st.warning("⚠️ Nenhuma collection encontrada. Crie uma na aba **Collections** e insira documentos.")
        st.stop()

    col_chat, col_cfg = st.columns([3, 1])
    with col_cfg:
        selected_label = st.selectbox("Collection", options, key="chat_collection")
        kind, selected_name = parse_selector(selected_label)
        if kind == "logical":
            st.caption("🔗 Lógica · two-stage (summary → chunks filtrados + RRF)")
        else:
            st.caption("🗄️ Legacy · busca direta na coll")

    # Histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-label">Você</div><div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label">🤖 Agente FII</div><div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("docs"):
                _render_docs_panel(msg["docs"])

    # Input
    user_input = st.chat_input("Pergunte sobre os relatórios de FIIs...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-label">Você</div><div class="chat-user">{user_input}</div>', unsafe_allow_html=True)

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
            except Exception as e:
                answer = f"❌ Erro ao processar: {e}"

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "docs": _serialize_docs(docs_used)}
        )
        st.markdown(f'<div class="chat-label">🤖 Agente FII</div><div class="chat-bot">{answer}</div>', unsafe_allow_html=True)
        if docs_used:
            _render_docs_panel(docs_used)

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
        logicals, legacies = classify_collections(client)
    except Exception as e:
        st.error(f"Não foi possível conectar ao Qdrant: {e}")
        st.stop()

    # ── Criar nova collection lógica ───────────
    non_summary_strategies = [s for s in ALL_STRATEGIES if s != "summary"]

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
                distance = st.selectbox("Métrica de distância", ["Cosine", "Dot", "Euclid"])
                selected_extras = st.multiselect(
                    "Estratégias de chunking",
                    options=non_summary_strategies,
                    default=non_summary_strategies,
                    help="`summary` é sempre incluída automaticamente.",
                )
            submit_create = st.form_submit_button("Criar coleção lógica", use_container_width=True)

        if submit_create:
            if not new_name.strip():
                st.error("Informe um nome para a coleção.")
            elif not CollectionNaming.is_valid_logical(new_name.strip()):
                st.error("Nome inválido. Use apenas letras, dígitos e `_`, máx 40 chars.")
            else:
                dist_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
                strategies = ["summary"] + list(selected_extras)
                try:
                    prov = get_provisioner(client)
                    result = prov.provision(
                        logical=new_name.strip(),
                        strategies=strategies,
                        hybrid=hybrid,
                        dense_dim=vector_size,
                        distance=dist_map[distance],
                    )
                    st.success(
                        f"✅ Coleção lógica **{new_name}** criada com "
                        f"{len(result['physical_names'])} físicas: "
                        f"{', '.join(result['physical_names'])}"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao provisionar: {e}")

    # ── Criar coll legada (single-coll) ────────
    with st.expander("➕ Criar coleção avulsa (legacy single-coll)", expanded=False):
        st.caption(
            "Modo simples: 1 vetor por coleção, sem multi-strategy. "
            "Mantido para compatibilidade — recomendamos coleções lógicas para projetos novos."
        )
        with st.form("create_legacy_form"):
            legacy_name = st.text_input("Nome", placeholder="ex: fii_legacy")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                legacy_size = st.selectbox(
                    "Dimensão", options=[768, 1024, 1536, 3072], index=3, key="legacy_size"
                )
            with col_b:
                legacy_distance = st.selectbox(
                    "Métrica", ["Cosine", "Dot", "Euclid"], key="legacy_distance"
                )
            with col_c:
                legacy_hybrid = st.toggle(
                    "Hybrid (BM25)", value=True, key="legacy_hybrid"
                )
            submit_legacy = st.form_submit_button("Criar (legacy)", use_container_width=True)

        if submit_legacy:
            if not legacy_name.strip():
                st.error("Informe um nome.")
            else:
                dist_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
                try:
                    repo = QdrantRepository(client)
                    repo.ensure_collection(
                        name=legacy_name.strip(),
                        dim=legacy_size,
                        distance=dist_map[legacy_distance],
                        hybrid=legacy_hybrid,
                    )
                    st.success(f"✅ Coleção legacy **{legacy_name}** criada.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao criar legacy: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Listagem das collections ───────────────
    total_count = len(logicals) + len(legacies)
    if total_count == 0:
        st.info("Nenhuma coleção encontrada. Crie uma acima.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(logicals)}</div><div class="metric-label">Lógicas</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(legacies)}</div><div class="metric-label">Legadas</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            total_pts = 0
            for li in logicals:
                for s in li["strategies"]:
                    try:
                        total_pts += client.get_collection(
                            CollectionNaming.to_physical(li["logical"], s)
                        ).points_count or 0
                    except Exception:
                        pass
            for n in legacies:
                try:
                    total_pts += client.get_collection(n).points_count or 0
                except Exception:
                    pass
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{total_pts}</div><div class="metric-label">Pontos totais</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Lógicas
        if logicals:
            st.markdown("### 🧩 Coleções lógicas")
            for li in logicals:
                logical = li["logical"]
                cfg = li["config"]
                hybrid_logical = cfg.get("hybrid", "—")
                dense_dim = cfg.get("dense_dim", "—")
                distance_label = str(cfg.get("distance", "—")).replace("Distance.", "")
                strategies_chips = " · ".join(li["strategies"])

                with st.expander(f"🧩 **{logical}** · {strategies_chips}", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Estratégias", len(li["strategies"]))
                    c2.metric("Dimensão", dense_dim)
                    c3.metric("Distância", distance_label)
                    c4.metric("Hybrid", "Sim" if hybrid_logical else "Não")

                    # Conta docs (pontos não-config) em __summary
                    summary_coll = CollectionNaming.to_physical(logical, "summary")
                    try:
                        from server.src.fii_rag.store import exclude_config_filter

                        n_docs = client.count(
                            collection_name=summary_coll,
                            count_filter=exclude_config_filter(),
                            exact=True,
                        ).count
                    except Exception:
                        n_docs = "—"

                    # Total de chunks somados em todas as strategies não-summary
                    n_chunks = 0
                    for s in li["strategies"]:
                        if s == "summary":
                            continue
                        try:
                            n_chunks += client.get_collection(
                                CollectionNaming.to_physical(logical, s)
                            ).points_count or 0
                        except Exception:
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
                                    st.success(f"Lógica **{logical}** deletada ({deleted} físicas apagadas).")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erro ao deletar: {e}")
                            else:
                                st.warning("Marque a confirmação para deletar.")

        # Legadas
        if legacies:
            st.markdown("### 🗄️ Coleções legadas")
            for name in legacies:
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
                                collection_name=name, limit=5, with_payload=True, with_vectors=False
                            )[0]
                            for p in points:
                                meta = p.payload or {}
                                content = meta.get("page_content", meta.get("text", "—"))
                                title = meta.get("metadata", {}).get("title", meta.get("title", "Sem título"))
                                keywords = meta.get("metadata", {}).get("keywords", meta.get("keywords", []))
                                st.markdown(f"**{title}**")
                                if keywords:
                                    st.caption(
                                        f"🏷️ {' · '.join(keywords[:5]) if isinstance(keywords, list) else keywords}"
                                    )
                                st.markdown(
                                    f'<div class="info-card" style="font-size:0.85rem;color:#94a3b8;">{content[:400]}{"..." if len(content) > 400 else ""}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro ao obter detalhes: {e}")

                    with st.form(f"delete_legacy_{name}"):
                        confirm = st.checkbox(f"Confirmo deletar legacy **{name}**")
                        if st.form_submit_button("🗑️ Deletar legacy", type="secondary"):
                            if confirm:
                                try:
                                    client.delete_collection(name)
                                    st.success(f"Legacy {name} deletada.")
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
        logicals, legacies = classify_collections(client)
    except Exception:
        logicals, legacies = [], []

    options = build_selector_options(logicals, legacies)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if options:
            selected_label = st.selectbox(
                "Coleção destino", options, key="ingest_collection"
            )
            target_kind, target_name = parse_selector(selected_label)
            if target_kind == "logical":
                st.caption(
                    "🔗 Coleção lógica · usa `IngestionPipeline` (multi-strategy + hybrid)"
                )
            else:
                st.caption("🗄️ Coleção legacy · usa pipeline antigo (single-coll)")
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

        # Opções avançadas (aplicáveis ao pipeline LEGACY)
        if target_kind == "legacy":
            with st.expander("⚙️ Opções avançadas (legacy)"):
                chunk_size = st.slider("Tamanho do chunk (tokens)", 200, 2000, 1000, 100)
                chunk_overlap = st.slider("Overlap do chunk (tokens)", 0, 500, 200, 50)
                extract_metadata = st.toggle("Extrair metadados via Mistral AI", value=True)
        else:
            st.info(
                "ℹ️ Pipeline lógico usa todas as estratégias da coleção e os "
                "parâmetros de chunking do `.env`. Os metadados-doc + sumário "
                "são extraídos automaticamente via DOC_SUMMARY_LLM."
            )
            chunk_size = chunk_overlap = None
            extract_metadata = True

        if st.button("🚀 Iniciar Ingestão", type="primary", use_container_width=True):
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            progress = st.progress(0, text="Preparando...")
            status = st.empty()

            try:
                if target_kind == "logical":
                    progress.progress(10, text="Inicializando pipeline...")
                    pipeline = get_ingestion_pipeline()
                    progress.progress(30, text=f"Ingerindo em {target_name} (multi-strategy)...")
                    result = pipeline.run(tmp_path, target_name)
                    progress.progress(100, text="Concluído!")
                    per_strategy = result["per_strategy"]
                    summary_lines = "\n".join(
                        f"• `{name}`: {n} pontos" + (" ⚠️" if n < 0 else "")
                        for name, n in per_strategy.items()
                    )
                    st.success(
                        f"✅ **{uploaded_file.name}** ingerido em **{target_name}**!\n\n"
                        f"doc_id: `{result['doc_id']}`\n\n{summary_lines}"
                    )
                else:
                    config = AppConfig()
                    llm, embed_model = config.get_llm_and_embeddings()

                    progress.progress(10, text="Conectando ao Qdrant...")
                    qdrant_provider = QdrantStoreProvider(url=config.qdrant_url, embed_model=embed_model)
                    vector_store = qdrant_provider.get_store(target_name)

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
                        progress.progress(
                            55, text=f"Extraindo metadados via Mistral ({len(splits)} chunks)..."
                        )
                        extractor = LangChainSemanticExtractor(llm=llm)
                        extractor_fn = extractor.get_extractors()
                        enriched_splits = extractor_fn(splits)
                    else:
                        enriched_splits = splits

                    progress.progress(80, text="Inserindo no Qdrant...")
                    vector_store.add_documents(documents=enriched_splits)
                    progress.progress(100, text="Concluído!")
                    st.success(
                        f"✅ **{uploaded_file.name}** ingerido (legacy)! "
                        f"{len(enriched_splits)} chunks na coll **{target_name}**."
                    )
            except Exception as e:
                st.error(f"❌ Erro durante a ingestão: {e}")
            finally:
                os.unlink(tmp_path)
