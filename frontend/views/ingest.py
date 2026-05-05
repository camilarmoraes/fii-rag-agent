"""Tela de inserção de PDF — dispatch entre pipeline lógico e legacy."""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from frontend.components.di import (
    get_ingestion_pipeline,
    get_raw_client,
)
from frontend.components.helpers import (
    build_selector_options,
    classify_collections,
    parse_selector,
)
from frontend.components.ui import render_hero
from server.src.fii_rag.chunking import (
    LangChainParser,
    LangChainSemanticExtractor,
)
from server.src.fii_rag.config import AppConfig
from server.src.fii_rag.db import QdrantStoreProvider


def render_ingest() -> None:
    render_hero(
        "Inserir Documento",
        "Faça upload de um PDF de relatório de FII para ingestão no Qdrant.",
    )

    try:
        client = get_raw_client()
        logicals, legacies = classify_collections(client)
    except Exception:  # noqa: BLE001
        logicals, legacies = [], []

    options = build_selector_options(logicals, legacies)
    if not options:
        st.warning(
            "⚠️ Nenhuma collection encontrada. "
            "Crie uma primeiro na aba **📚 Collections**."
        )
        st.stop()

    target_kind, target_name = _render_target_selector(options)
    uploaded_file = st.file_uploader(
        "Selecione o arquivo PDF do relatório de FII",
        type=["pdf"],
        help="Apenas arquivos PDF são suportados.",
    )

    if uploaded_file is None:
        return

    st.markdown(
        f'<div class="info-card">📄 <b>{uploaded_file.name}</b> · '
        f"{uploaded_file.size / 1024:.1f} KB</div>",
        unsafe_allow_html=True,
    )

    chunk_size, chunk_overlap, extract_metadata = _render_advanced_options(target_kind)

    if st.button("🚀 Iniciar Ingestão", type="primary", use_container_width=True):
        _run_ingestion(
            uploaded_file=uploaded_file,
            target_kind=target_kind,
            target_name=target_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_metadata=extract_metadata,
        )


def _render_target_selector(options: list[str]) -> tuple[str, str]:
    col_a, _ = st.columns([2, 1])
    with col_a:
        selected_label = st.selectbox(
            "Coleção destino", options, key="ingest_collection"
        )
        kind, name = parse_selector(selected_label)
        if kind == "logical":
            st.caption(
                "🔗 Coleção lógica · usa `IngestionPipeline` (multi-strategy + hybrid)"
            )
        else:
            st.caption("🗄️ Coleção legacy · usa pipeline antigo (single-coll)")
    return kind, name


def _render_advanced_options(target_kind: str):
    if target_kind == "legacy":
        with st.expander("⚙️ Opções avançadas (legacy)"):
            chunk_size = st.slider("Tamanho do chunk (tokens)", 200, 2000, 1000, 100)
            chunk_overlap = st.slider("Overlap do chunk (tokens)", 0, 500, 200, 50)
            extract_metadata = st.toggle(
                "Extrair metadados via Mistral AI", value=True
            )
        return chunk_size, chunk_overlap, extract_metadata

    st.info(
        "ℹ️ Pipeline lógico usa todas as estratégias da coleção e os "
        "parâmetros de chunking do `.env`. Os metadados-doc + sumário "
        "são extraídos automaticamente via DOC_SUMMARY_LLM."
    )
    return None, None, True


def _run_ingestion(
    uploaded_file,
    target_kind: str,
    target_name: str,
    chunk_size,
    chunk_overlap,
    extract_metadata: bool,
) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    progress = st.progress(0, text="Preparando...")
    status = st.empty()

    try:
        if target_kind == "logical":
            _ingest_logical(uploaded_file.name, tmp_path, target_name, progress)
        else:
            _ingest_legacy(
                uploaded_file=uploaded_file,
                tmp_path=tmp_path,
                target_name=target_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                extract_metadata=extract_metadata,
                progress=progress,
                status=status,
            )
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Erro durante a ingestão: {e}")
    finally:
        os.unlink(tmp_path)


def _ingest_logical(filename: str, tmp_path: str, logical: str, progress) -> None:
    progress.progress(10, text="Inicializando pipeline...")
    pipeline = get_ingestion_pipeline()
    progress.progress(30, text=f"Ingerindo em {logical} (multi-strategy)...")
    result = pipeline.run(tmp_path, logical)
    progress.progress(100, text="Concluído!")
    per_strategy = result["per_strategy"]
    summary_lines = "\n".join(
        f"• `{name}`: {n} pontos" + (" ⚠️" if n < 0 else "")
        for name, n in per_strategy.items()
    )
    st.success(
        f"✅ **{filename}** ingerido em **{logical}**!\n\n"
        f"doc_id: `{result['doc_id']}`\n\n{summary_lines}"
    )


def _ingest_legacy(
    uploaded_file,
    tmp_path: str,
    target_name: str,
    chunk_size: int,
    chunk_overlap: int,
    extract_metadata: bool,
    progress,
    status,
) -> None:
    config = AppConfig()
    llm, embed_model = config.get_llm_and_embeddings()

    progress.progress(10, text="Conectando ao Qdrant...")
    provider = QdrantStoreProvider(url=config.qdrant_url, embed_model=embed_model)
    vector_store = provider.get_store(target_name)

    progress.progress(25, text="Carregando PDF...")
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()
    status.info(f"📄 {len(documents)} página(s) carregada(s).")

    progress.progress(40, text="Fragmentando em chunks...")
    parser = LangChainParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = parser.get_parser().split_documents(documents)
    status.info(f"✂️ {len(splits)} chunks gerados.")

    if extract_metadata:
        progress.progress(
            55, text=f"Extraindo metadados via Mistral ({len(splits)} chunks)..."
        )
        extractor_fn = LangChainSemanticExtractor(llm=llm).get_extractors()
        enriched = extractor_fn(splits)
    else:
        enriched = splits

    progress.progress(80, text="Inserindo no Qdrant...")
    vector_store.add_documents(documents=enriched)
    progress.progress(100, text="Concluído!")
    st.success(
        f"✅ **{uploaded_file.name}** ingerido (legacy)! "
        f"{len(enriched)} chunks na coll **{target_name}**."
    )
