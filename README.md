# 🏢 FII RAG Agent

> **Agente de IA especializado em análise de Fundos Imobiliários Brasileiros (FIIs)**, construído com Qdrant nativo (hybrid search BM25 + dense), LangChain (LLMs), pydantic-settings e Streamlit.

---

## Visão Geral

RAG sobre relatórios gerenciais de FIIs com pipeline em duas fases:

1. **Ingestão multi-strategy**: cada PDF é processado por **5 estratégias de chunking distintas em paralelo** (`summary`, `recursive`, `semantic`, `doc_aware`, `hierarchical`) — cada uma vai para sua própria coleção física no Qdrant. O usuário cria uma "coleção lógica" `fii_2026` e o sistema provisiona internamente `fii_2026__summary`, `fii_2026__recursive`, etc.
2. **Two-stage retrieval**: a query primeiro busca em `__summary` para identificar os documentos mais relevantes (top-K `doc_id`s), depois busca em paralelo nas demais sub-collections com `Filter(doc_id IN [...])`, funde os rankings via RRF nativo do Qdrant, e re-rankeia o top final via cross-encoder.

---

## Arquitetura

```
                         ┌────────────────┐
PDF ──► PdfLoader ──────►│LoadedDocument  │ (texto + páginas + markdown)
                         └────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────┐
                │ DocMetadataExtractor (1 LLM call)│
                │ → ticker, CNPJ, datas, summary   │
                └──────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────────────┐
              ▼                 ▼                          ▼
         ┌────────┐         ┌─────────┐              ┌────────────┐
         │summary │         │recursive│   …          │hierarchical│  (paralelo)
         └────────┘         └─────────┘              └────────────┘
              │                 │                          │
              ▼                 ▼                          ▼
        ChunkEnricher (LLM batch + numerics regex/LLM fallback)
              │                 │                          │
              ▼                 ▼                          ▼
        DenseEmbedder + BM25SparseEmbedder
              │                 │                          │
              ▼                 ▼                          ▼
   fii_2026__summary  fii_2026__recursive      fii_2026__hierarchical
   (1 ponto/doc)      (chunks)                  (filhos com parent_text)
```

**Query (two-stage)**:

```
                    Query
                      │
                      ▼
   Stage 1: query_dense em __summary  ──► doc_ids relevantes
                      │
                      ▼
   Stage 2: ThreadPoolExecutor ─► query_hybrid em N strategies
                      │           Filter(doc_id IN [...])
                      ▼
                rrf_merge (fusão RRF + dedup)
                      │
                      ▼
              CrossEncoderReranker (top-30 → top-5)
                      │
                      ▼
              create_stuff_documents_chain → Mistral LLM
```

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| **Vector DB** | Qdrant (puro, via `qdrant-client` — sem `langchain-qdrant`) |
| **Hybrid search** | dense (Google `text-embedding-004`) + sparse (`Qdrant/bm25` via FastEmbed) com fusão RRF nativa |
| **Reranker** | `BAAI/bge-reranker-base` via `sentence-transformers` direto, ou `LLMReranker` (alternativa) |
| **Chunking** | 5 estratégias: `RecursiveCharacterTextSplitter`, `SemanticChunker` (langchain_experimental), `DocAwareChunker` (heading-aware via `pymupdf4llm`), `HierarchicalChunker` (parent/child), `DocumentSummaryChunker` |
| **LLM por estágio** | `CHAT` / `DOC_SUMMARY` / `CHUNK_METADATA` / `RERANK` — cada um configurável em provider/modelo via `.env` (mistral / google / openai) |
| **Config** | `pydantic-settings` + `python-dotenv` |
| **PDF Parser** | `PyMuPDFLoader` + `pymupdf4llm` (markdown estruturado) |
| **Frontend** | Streamlit modular (`frontend/views/` + `frontend/components/`) |

---

## Estrutura do Projeto

```
fii-rag-agent/
├── 📄 main.py                       # Entry point CLI (--ingest / --chat) — modo legacy
├── 📄 docker-compose.yml            # Sobe Qdrant local (:6333 / :6334)
├── 📄 pyproject.toml / uv.lock      # Dependências (uv)
├── 📄 .env.example                  # Template de envs
├── 📁 data/                         # PDFs de relatórios (gitignored)
├── 📁 qdrant_data/                  # Volume persistido (gitignored)
│
├── 📁 server/src/fii_rag/
│   ├── interfaces.py                # ABCs (legadas + IDenseEmbedder/ISparseEmbedder/ILLMFactory)
│   ├── config.py                    # AppConfig + pydantic-settings (8 seções)
│   ├── agent.py                     # RAGAgent: set_logical/legacy_collection
│   │
│   ├── schemas/                     # DocumentMetadata, ChunkMetadata, ChunkPayload, NumericFacts
│   ├── llm/                         # LLMStage enum + LLMFactory.for_stage()
│   ├── embeddings/                  # DenseEmbedderFactory + BM25SparseEmbedder
│   │
│   ├── store/
│   │   ├── client.py                # QdrantClientFactory (cacheado por URL)
│   │   ├── repository.py            # QdrantRepository (upsert/query_dense/query_hybrid)
│   │   ├── schema.py                # CollectionSchemaBuilder (named "dense" + sparse)
│   │   ├── naming.py                # CollectionNaming (logical ↔ physical)
│   │   └── provisioner.py           # LogicalCollectionProvisioner (cria N físicas + payload indexes + _LOGICAL_CONFIG_)
│   │
│   ├── chunking/                    # 5 estratégias + base (IChunkingStrategy) + registry
│   ├── extraction/                  # DocMetadataExtractor / ChunkMetadataExtractor / NumericFactsExtractor
│   ├── ingestion/                   # PdfLoader + ChunkEnricher + IngestionPipeline (paralelo por strategy)
│   │
│   ├── retrieval/
│   │   ├── base.py                  # IRetriever + RetrievalResult
│   │   ├── single_strategy.py       # SingleStrategyRetriever
│   │   ├── two_stage.py             # TwoStageRetriever (summary → doc_ids → fusion → rerank)
│   │   ├── fusion.py                # rrf_merge + dedup
│   │   ├── builder.py               # RetrieverBuilder.build_for_logical/legacy
│   │   ├── lc_adapter.py            # LangChainRetrieverAdapter (envolve IRetriever em BaseRetriever)
│   │   └── rerank/
│   │       ├── cross_encoder.py     # via sentence-transformers direto
│   │       └── llm_reranker.py      # alternativa via RERANK_LLM_MODEL
│   │
│   └── (db.py / retriever.py / chunking/legacy.py / ingestion/legacy.py
│        — shims preservados para coleções legadas no frontend)
│
└── 📁 frontend/
    ├── app.py                       # Entrypoint: page_config + sidebar + dispatcher
    ├── styles.css                   # CSS extraído
    ├── components/
    │   ├── di.py                    # Caches @st.cache_resource
    │   ├── helpers.py               # classify_collections, parse_selector
    │   └── ui.py                    # inject_styles, render_hero, render_docs_panel
    └── views/                       # ("pages" é reservado pelo Streamlit)
        ├── chat.py
        ├── collections.py
        └── ingest.py
```

---

## Pré-requisitos

- **Python** ≥ 3.13
- **Docker** + **Docker Compose** (para o Qdrant)
- **uv** (gerenciador de pacotes recomendado)
- **MISTRAL_API_KEY** + **GOOGLE_API_KEY** (Mistral AI + Google AI Studio)

---

## Instalação

```bash
git clone https://github.com/camilarmoraes/fii-rag-agent.git
cd fii-rag-agent

uv sync                       # cria .venv e instala deps
docker compose up -d          # sobe Qdrant em :6333 (REST) / :6334 (gRPC)
cp .env.example .env          # preenche MISTRAL_API_KEY e GOOGLE_API_KEY
```

Edite o `.env` com suas credenciais. Os defaults (modelo, temperatura, top-K) já estão razoáveis e podem ser ajustados depois.

---

## Configuração (.env)

Variáveis principais (todas opcionais exceto API keys):

```env
# Provedores
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...

# Qdrant
QDRANT_URL=http://localhost:6333

# Embeddings
DENSE_EMBED_PROVIDER=google           # google | mistral | openai
DENSE_EMBED_MODEL=text-embedding-004
DENSE_EMBED_DIM=768
SPARSE_EMBED_MODEL=Qdrant/bm25

# LLM por estágio (cada um configurável em provider/modelo)
CHAT_LLM_PROVIDER=mistral
CHAT_LLM_MODEL=mistral-large-latest
DOC_SUMMARY_LLM_MODEL=mistral-large-latest
CHUNK_METADATA_LLM_MODEL=mistral-small-latest    # mais barato p/ alta volumetria
RERANK_BACKEND=cross_encoder                      # cross_encoder | llm

# Pipeline
INGEST_MAX_WORKERS=4
RECURSIVE_CHUNK_SIZE=1000
HIERARCHICAL_PARENT_SIZE=2000
HIERARCHICAL_CHILD_SIZE=400

# Retrieval
STAGE1_TOP_K=8
STAGE2_TOP_K_PER_STRATEGY=20
RRF_K=60
FINAL_TOP_K=5
```

Lista completa em `.env.example`.

---

## Uso

### Frontend (Streamlit)

```bash
streamlit run frontend/app.py    # rode da raiz do projeto
```

Abre em `http://localhost:8501`:

- **💬 Chat** — selecione uma coleção (lógica ou legacy), faça perguntas, veja painel "📚 Trechos usados (N)" com badges (`#i · strategy · ticker · YYYYTQ · p.N · § heading`)
- **📚 Collections** — crie lógicas multi-strategy (multi-select), legadas single-coll; lista em duas seções (🧩 Lógicas / 🗄️ Legadas); deletar lógica apaga as 5 físicas em cascata
- **📄 Inserir Documento** — selectbox unificado `[L] fii_2026` / `[Legacy] fii_abril`; lógica usa `IngestionPipeline` (paralelo, todas as strategies); legacy usa `PDFIngestionManager`

### CLI (legado)

```bash
python main.py --ingest    # ingere data/*.pdf na coleção `fii_reports`
python main.py --chat      # REPL terminal
```

> CLI usa o pipeline legacy (single-coll); para a arquitetura nova, use o frontend.

---

## Decisões de design

- **Qdrant nativo** sobre `langchain-qdrant`: ganhos diretos em `PointStruct`/`upsert`, named vectors, sparse vectors, prefetch+fusion=RRF, payload indexes e filtros avançados
- **Multi-strategy física**: cada estratégia em sua sub-collection — `recursive` para baseline rápido, `semantic` para coerência, `doc_aware` para preservar headings, `hierarchical` para precisão fina + contexto largo, `summary` como roteador no stage 1
- **Two-stage routing** (em vez de RRF direto entre todas): o `summary` filtra os `doc_id`s no stage 1, então o stage 2 evita misturar trechos de docs irrelevantes; ganho grande em queries que mencionam ticker/período específico
- **Hidratação hierarchical**: o filho indexado é pequeno (precisão na busca); o LLM final recebe `parent_text + child` (contexto largo)
- **Coleções legadas preservadas**: chat e ingest aceitam ambas; migração gradual sem perda de dados
- **LLMs por estágio**: `CHUNK_METADATA` é o estágio mais caro em volume — recomendado `mistral-small-latest`; `CHAT` pode usar `mistral-large-latest`
