# FII RAG Agent — Frontend (Streamlit)

> Interface web para interagir com o **FII RAG Agent** de forma visual e intuitiva.

---

## 📋 Funcionalidades

| Tela | Descrição |
|---|---|
| **💬 Chat** | Converse com o agente sobre os relatórios de FIIs ingeridos |
| **📚 Collections** | Crie, visualize e delete collections do Qdrant; explore os documentos armazenados |
| **📄 Inserir Documento** | Faça upload de PDFs e dispare o pipeline de ingestão diretamente pela UI |

---

## 🗂️ Estrutura

```
frontend/
├── app.py       # Aplicação Streamlit completa (todas as telas)
└── README.md    # Esta documentação
```

O frontend usa um único arquivo `app.py` com navegação via sidebar, mantendo o código simples e direto.

---

## Pré-requisitos

- Todos os pré-requisitos do projeto raiz já instalados (veja o [README principal](../README.md))
- **Streamlit** instalado:

```bash
pip install streamlit
# ou com uv:
uv add streamlit
```

- Qdrant rodando via Docker:

```bash
docker-compose up -d
```

- Arquivo `.env` configurado na raiz do projeto (o frontend lê as variáveis de lá automaticamente):

```env
MISTRAL_API_KEY="..."
GOOGLE_API_KEY="..."
QDRANT_URL="http://localhost:6333"
```

---

## Como Rodar

Execute a partir da **raiz do projeto** (não de dentro de `frontend/`), para que os imports do pacote `server` funcionem:

```bash
streamlit run frontend/app.py
```

O Streamlit abrirá automaticamente no navegador em:

```
http://localhost:8501
```

---

## Telas

### 💬 Chat

- Selecione a collection que deseja consultar
- Digite sua pergunta no campo de chat
- O agente busca os chunks mais relevantes via **busca vetorial + reranking BAAI** e gera a resposta com **Mistral AI**
- O histórico da conversa é mantido na sessão
- Botão para limpar o histórico

### 📚 Collections

- **Status do Qdrant** visível na sidebar (verde/vermelho)
- **Criar collection**: formulário com nome, dimensão do vetor e métrica de distância
- **Listar collections**: métricas globais (total de collections e documentos) + cards expandíveis por collection
- Cada card exibe: número de documentos, dimensão vetorial, métrica de distância e **amostra de até 5 documentos** com título, palavras-chave e preview do conteúdo
- **Deletar collection** com checkbox de confirmação

### 📄 Inserir Documento

- Selecione a collection destino (ou informe o nome para criar uma nova)
- Faça upload de um arquivo PDF
- Configure opções avançadas de chunking (tamanho e overlap)
- Toggle para ativar/desativar extração de metadados semânticos via Mistral AI
- Barra de progresso em tempo real durante a ingestão

---

## Detalhes Técnicos

### Caching com `@st.cache_resource`

Os componentes pesados (modelos, conexões) são inicializados uma única vez e cacheados pela sessão do Streamlit via `@st.cache_resource`, evitando recarregamento a cada interação.

### Injeção de Dependências

O frontend reutiliza exatamente os mesmos módulos do backend (`server/src/fii_rag/`), sem duplicação de lógica. A função `get_components()` monta o container de dependências igual ao `main.py`.

### Seleção de Collection no Chat

Ao trocar a collection selecionada no chat, o agente reseta a chain (`agent.chain = None`) para reconstruir o retriever apontando para a nova collection.

---

## Solução de Problemas

| Problema | Solução |
|---|---|
| `Qdrant offline` na sidebar | Verifique se `docker-compose up -d` foi executado |
| Erro de imports ao rodar | Execute `streamlit run frontend/app.py` da **raiz** do projeto |
| Metadados não aparecem nos documentos | Os PDFs foram ingeridos sem extração de metadados; re-ingira com a opção ativada |
| Erro de API Key | Verifique o `.env` na raiz do projeto |
