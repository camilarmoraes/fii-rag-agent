"""Dependency injection: instâncias cacheadas dos componentes do backend."""

from __future__ import annotations

import streamlit as st
from qdrant_client import QdrantClient

from server.src.fii_rag.agent import RAGAgent
from server.src.fii_rag.chunking import LangChainParser, LangChainSemanticExtractor
from server.src.fii_rag.config import AppConfig
from server.src.fii_rag.db import QdrantStoreProvider
from server.src.fii_rag.ingestion import (
    IngestionPipeline,
    PDFIngestionManager,
    build_ingestion_pipeline,
)
from server.src.fii_rag.retrieval import RetrieverBuilder, build_retriever_builder
from server.src.fii_rag.retriever import HybridQueryEngineBuilder
from server.src.fii_rag.store import LogicalCollectionProvisioner, QdrantRepository


@st.cache_resource(show_spinner="⚙️ Inicializando componentes...")
def get_components():
    """Componentes legados — usados pelo chat (RAGAgent) e ingest legacy."""
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
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def get_raw_client() -> QdrantClient:
    """Retorna um `QdrantClient` novo a cada chamada — cheap o suficiente."""
    return QdrantClient(url=AppConfig().qdrant_url)


def get_provisioner(client: QdrantClient) -> LogicalCollectionProvisioner:
    return LogicalCollectionProvisioner(QdrantRepository(client))


@st.cache_resource(show_spinner="⚙️ Inicializando pipeline de ingestão...")
def get_ingestion_pipeline() -> IngestionPipeline:
    return build_ingestion_pipeline(AppConfig())


@st.cache_resource(show_spinner="⚙️ Inicializando retriever builder...")
def get_retriever_builder() -> RetrieverBuilder:
    return build_retriever_builder(AppConfig())
