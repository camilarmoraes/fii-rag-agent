# 🏢 FII RAG Agent

> **Agente de Inteligência Artificial especializado em análise de Fundos Imobiliários Brasileiros (FIIs)**, construído com LangChain, Qdrant, Mistral AI e Google Embeddings.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Como Usar (CLI)](#como-usar-cli)
- [Módulos e Componentes](#módulos-e-componentes)
- [Princípios de Design (SOLID)](#princípios-de-design-solid)
- [Fluxo de Dados](#fluxo-de-dados)

---

## Visão Geral

O **FII RAG Agent** é um sistema de **Retrieval-Augmented Generation (RAG)** desenvolvido para responder perguntas sobre relatórios de Fundos de Investimento Imobiliário (FIIs) com base em documentos PDF ingeridos pelo próprio sistema.

O fluxo é dividido em duas etapas principais:

1. **Ingestão**: os PDFs são carregados, fragmentados em chunks semânticos, enriquecidos com metadados extraídos via Mistral AI e persistidos no banco vetorial Qdrant.
2. **Chat**: o usuário faz perguntas em linguagem natural e o agente recupera os trechos mais relevantes via busca vetorial + reranking neural, gerando respostas com o LLM Mistral.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         FII RAG Agent                           │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   PDF Files  │───▶│  Ingestion   │───▶│   Qdrant Vector  │   │
│  │  (data/)    │    │   Pipeline   │    │     Database     │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                            │                       │            │
│                     ┌──────▼──────┐        ┌───────▼──────┐    │
│                     │  Mistral AI │        │  Hybrid      │    │
│                     │  (Metadata  │        │  Retriever + │    │
│                     │  Extractor) │        │  BAAI Rerank │    │
│                     └─────────────┘        └───────┬──────┘    │
│                                                    │            │
│                     ┌──────────────────────────────▼─────────┐ │
│                     │          RAG Agent (LCEL Chain)         │ │
│                     │  Retrieval Chain + Mistral LLM Answer   │ │
│                     └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline de Ingestão

```
PDF → PyMuPDF Loader → RecursiveCharacterTextSplitter → Mistral (Metadata) → Google Embeddings → Qdrant
```

### Pipeline de Consulta

```
Query → Google Embeddings → Qdrant (Top-K) → BAAI Reranker (Top-N) → Mistral LLM → Resposta
```

---

## Tecnologias

| Componente | Tecnologia | Função |
|---|---|---|
| **LLM** | Mistral AI (`mistral-small-latest`) | Geração de respostas e extração de metadados |
| **Embeddings** | Google Generative AI (`text-embedding-004`) | Vetorização semântica dos chunks |
| **Vector DB** | Qdrant | Armazenamento e busca por similaridade vetorial |
| **Reranker** | BAAI `bge-reranker-base` (HuggingFace) | Re-ordenação neural dos resultados recuperados |
| **Framework** | LangChain + LangChain Classic | Orquestração de chains e retrievers |
| **PDF Parser** | PyMuPDF (`langchain-community`) | Carregamento de documentos PDF |
| **Infraestrutura** | Docker + Docker Compose | Execução do Qdrant localmente |
| **Gerenciador de deps** | `uv` | Gerenciamento de dependências Python rápido |

---

## Estrutura do Projeto

```
fii-rag-agent/
│
├── 📄 main.py                    # Entry point CLI (ingestão e chat)
├── 📄 pyproject.toml             # Dependências e configurações do projeto (uv)
├── 📄 requirements.txt           # Dependências legadas (referência)
├── 📄 docker-compose.yml         # Subida do Qdrant via Docker
├── 📄 .env.example               # Template de variáveis de ambiente
├── 📁 data/                      # Diretório para PDFs de FIIs
├── 📁 qdrant_data/               # Volume persistido do banco vetorial
│
└── 📁 server/
    └── 📁 src/
        └── 📁 fii_rag/           # Pacote principal
            ├── __init__.py
            ├── interfaces.py     # Contratos abstratos (ABCs / SOLID)
            ├── config.py         # Configuração centralizada (env vars + modelos)
            ├── db.py             # Provider Qdrant (vector store)
            ├── chunking.py       # Particionamento e extração de metadados
            ├── ingestion.py      # Orquestrador do pipeline de ingestão
            ├── retriever.py      # Builder do retriever híbrido com reranking
            └── agent.py          # Agente RAG LCEL (chain + chat loop)
```

---

## Pré-requisitos

- **Python** >= 3.13
- **Docker** e **Docker Compose** (para o Qdrant)
- **uv** (gerenciador de pacotes recomendado)
- Conta no **Mistral AI** com API Key
- Conta no **Google AI Studio** com API Key

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/fii-rag-agent.git
cd fii-rag-agent
```

### 2. Instale as dependências com `uv`

```bash
# Instalar uv (caso não tenha)
pip install uv

# Criar ambiente virtual e instalar dependências
uv sync
```

> Alternativamente com pip:
> ```bash
> pip install -r requirements.txt
> ```

### 3. Suba o Qdrant via Docker

```bash
docker-compose up -d
```

Isso inicia o Qdrant nas portas:
- `6333` → REST API
- `6334` → gRPC API

O banco de dados é persistido no diretório `qdrant_data/`.

---

## Configuração

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# Chaves de API necessárias
MISTRAL_API_KEY="sua_chave_da_mistral_aqui"
GOOGLE_API_KEY="sua_chave_do_google_ai_studio_aqui"

# URL do Qdrant (padrão Docker local)
QDRANT_URL="http://localhost:6333"

# Modelo Mistral (opcional, padrão: mistral-small-latest)
MISTRAL_MODEL="mistral-small-latest"
```

### Obtendo as API Keys

- **Mistral AI**: [https://console.mistral.ai/](https://console.mistral.ai/)
- **Google AI Studio**: [https://aistudio.google.com/](https://aistudio.google.com/)

---

## Como Usar (CLI)

### Ingestão de PDFs

1. Coloque seus PDFs de relatórios de FIIs na pasta `data/`:
   ```
   data/
   ├── relatorio-fii-xpto-2024.pdf
   └── fundo-abc-informe-2024.pdf
   ```

2. Execute a ingestão:
   ```bash
   python main.py --ingest
   ```

   O processo irá:
   - Carregar todos os PDFs da pasta `data/`
   - Fragmentar cada documento em chunks de 1000 tokens com overlap de 200
   - Extrair metadados semânticos via Mistral (título, palavras-chave, resumo)
   - Gerar embeddings via Google `text-embedding-004`
   - Persistir no Qdrant na coleção `fii_reports`

### Chat Interativo

```bash
python main.py --chat
```

O agente iniciará um loop de chat no terminal. Exemplos de perguntas:

```
👤 Você: Qual é o dividend yield do XPML11?
👤 Você: Me fale sobre os ativos do KNRI11.
👤 Você: Quais são os riscos mencionados nos relatórios?
```

Para sair, digite `sair`, `exit` ou `quit`, ou pressione `Ctrl+C`.

---

## Módulos e Componentes

### `interfaces.py` — Contratos Abstratos

Define os contratos (interfaces) que todas as implementações devem seguir. Garante inversão de dependências (princípio DIP do SOLID):

| Interface | Responsabilidade |
|---|---|
| `IVectorStoreProvider` | Prover acesso ao banco vetorial |
| `IDocumentParser` | Definir regras de chunking |
| `IMetadataExtractor` | Extrair metadados dos chunks |
| `IQueryEngineBuilder` | Construir o motor de busca/retriever |

---

### `config.py` — Configuração

Classe `AppConfig`: carrega variáveis de ambiente com `python-dotenv` e instancia os modelos LangChain:
- `ChatMistralAI` (LLM)
- `GoogleGenerativeAIEmbeddings` (embeddings)

---

### `db.py` — Banco Vetorial

Classe `QdrantStoreProvider`: implementa `IVectorStoreProvider`.

- Conecta ao Qdrant via `QdrantClient`
- Cria a coleção automaticamente se não existir (com dimensão detectada automaticamente via embedding de teste)
- Retorna um `QdrantVectorStore` pronto para uso pelo LangChain

---

### `chunking.py` — Chunking e Metadados

**`LangChainParser`** (implementa `IDocumentParser`):
- Usa `RecursiveCharacterTextSplitter` do LangChain
- Parâmetros padrão: `chunk_size=1000`, `chunk_overlap=200`

**`LangChainSemanticExtractor`** (implementa `IMetadataExtractor`):
- Usa Mistral AI com saída estruturada (`with_structured_output`)
- Extrai para cada chunk: `title`, `keywords` (≥5), `summary`
- Schema Pydantic: `ExtractedMetadata`

---

### `ingestion.py` — Pipeline de Ingestão

Classe `PDFIngestionManager`: orquestra o pipeline completo:

1. Lista todos os PDFs em `data/`
2. Carrega com `PyMuPDFLoader`
3. Fragmenta com o parser (`RecursiveCharacterTextSplitter`)
4. Enriquece com metadados semânticos (Mistral)
5. Persiste no Qdrant via `vector_store.add_documents()`

---

### `retriever.py` — Retriever Híbrido

Classe `HybridQueryEngineBuilder` (implementa `IQueryEngineBuilder`):

1. Obtém o `QdrantVectorStore` via provider
2. Cria um `base_retriever` com top-k=10 por similaridade cosseno
3. Aplica Cross-Encoder Reranking com `BAAI/bge-reranker-base` (multilingual)
4. Retorna um `ContextualCompressionRetriever` com top-n=3 resultados finais
5. Fallback para o retriever base caso `sentence-transformers` não esteja disponível

---

### `agent.py` — Agente RAG

Classe `RAGAgent`: orquestra o chat completo via LCEL (LangChain Expression Language):

1. Constrói o retriever via `HybridQueryEngineBuilder`
2. Cria o prompt de sistema especializado em FIIs
3. Monta a chain: `create_stuff_documents_chain` + `create_retrieval_chain`
4. Executa o loop de chat no terminal, processando respostas via `chain.invoke()`

---

## Princípios de Design (SOLID)

O projeto foi desenvolvido seguindo os princípios SOLID:

| Princípio | Onde se aplica |
|---|---|
| **S** — Single Responsibility | Cada classe tem uma única responsabilidade (`AppConfig`, `PDFIngestionManager`, `RAGAgent`) |
| **O** — Open/Closed | Novas implementações de parser ou retriever podem ser adicionadas sem alterar código existente |
| **L** — Liskov Substitution | Qualquer implementação das ABCs pode substituir outra sem quebrar o sistema |
| **I** — Interface Segregation | Interfaces pequenas e focadas (`IDocumentParser`, `IMetadataExtractor`, etc.) |
| **D** — Dependency Inversion | `PDFIngestionManager` e `RAGAgent` dependem de abstrações, não de implementações concretas |

---

## Fluxo de Dados

```
┌─────────────┐
│  PDF Files   │
└──────┬──────┘
       │ PyMuPDFLoader
       ▼
┌─────────────────────────────────┐
│  List[Document] (LangChain)     │
└──────┬──────────────────────────┘
       │ RecursiveCharacterTextSplitter
       ▼
┌─────────────────────────────────┐
│  List[Document] (chunks)        │
│  chunk_size=1000, overlap=200   │
└──────┬──────────────────────────┘
       │ Mistral AI (structured_output)
       ▼
┌─────────────────────────────────┐
│  List[Document] (enriched)      │
│  + metadata: title, keywords,   │
│    summary                      │
└──────┬──────────────────────────┘
       │ Google text-embedding-004
       ▼
┌─────────────────────────────────┐
│  Qdrant Vector Store            │
│  Collection: fii_reports        │
│  Distance: COSINE               │
└─────────────────────────────────┘
       │
       │ (Query time)
       ▼
┌─────────────────────────────────┐
│  Base Retriever (top_k=10)      │
└──────┬──────────────────────────┘
       │ BAAI/bge-reranker-base
       ▼
┌─────────────────────────────────┐
│  Reranked Docs (top_n=3)        │
└──────┬──────────────────────────┘
       │ create_stuff_documents_chain
       ▼
┌─────────────────────────────────┐
│  Mistral LLM → Answer           │
└─────────────────────────────────┘
```

---

## 🖥️ Frontend (Streamlit)

O projeto conta com uma interface web interativa construída com Streamlit, disponível na pasta `frontend/`.

Acesse a documentação completa do frontend em: [`frontend/README.md`](frontend/README.md)

```bash
# Rodar o frontend
cd frontend
streamlit run app.py
```

---

## 📝 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
