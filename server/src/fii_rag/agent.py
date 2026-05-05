"""`RAGAgent` — controlador do chat com retriever plugável.

Modos de uso:

1. **Legado** (compat com `main.py` e código antigo):
       agent.collection_name = "fii_reports"
       agent.chain = None
       agent._setup_engine()
   Usa o `query_engine_builder` (`HybridQueryEngineBuilder`) injetado no init.

2. **Novo** (usado pelo frontend após PR 5):
       agent.set_logical_collection("fii_2026", retriever_builder)
       agent.set_legacy_collection("fii_abril", retriever_builder)
   Constrói um `IRetriever` (TwoStage ou Single) e o embrulha em
   `LangChainRetrieverAdapter` para preservar o `create_retrieval_chain`.

Após qualquer set_*, `chain.invoke({"input": ...})` retorna
`{"input": ..., "context": [Document, ...], "answer": "..."}`. O frontend lê
`response["context"]` para exibir o painel "Trechos usados" — cada `Document`
tem na metadata as chaves `_score`, `_strategy`, `_chunk_id` + payload do Qdrant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from server.src.fii_rag.interfaces import IQueryEngineBuilder
from server.src.fii_rag.retrieval.lc_adapter import LangChainRetrieverAdapter

if TYPE_CHECKING:
    from server.src.fii_rag.retrieval.builder import RetrieverBuilder

SYSTEM_PROMPT = (
    "Você é um assistente sênior especializado em Fundos Imobiliários "
    "Brasileiros (FIIs). Use os seguintes trechos de contexto recuperado para "
    "responder à pergunta. Se você não souber a resposta, seja honesto e diga "
    "que não sabe. Responda de forma clara e estruturada, citando ticker, "
    "período e valores quando relevante.\n\n"
    "Aqui está o contexto extraído dos relatórios:\n"
    "{context}"
)


class RAGAgent:
    def __init__(self, query_engine_builder: IQueryEngineBuilder, llm: Any):
        self.query_engine_builder = query_engine_builder
        self.llm = llm
        self.chain = None
        self.collection_name = "fii_reports"
        self._mode: str = "legacy_builder"

    # ------------------------------------------------------------------
    # Modo legado (mantido para back-compat)
    # ------------------------------------------------------------------

    def _setup_engine(self) -> None:
        if not self.chain:
            retriever = self.query_engine_builder.build(self.collection_name)
            self._build_chain_from_lc_retriever(retriever)
            self._mode = "legacy_builder"

    # ------------------------------------------------------------------
    # Modo novo (RetrieverBuilder + IRetriever)
    # ------------------------------------------------------------------

    def set_logical_collection(
        self,
        logical: str,
        retriever_builder: "RetrieverBuilder",
        top_k: Optional[int] = None,
    ) -> None:
        irretriever = retriever_builder.build_for_logical(logical)
        self._activate_irretriever(irretriever, top_k or _resolve_top_k(retriever_builder))
        self.collection_name = logical
        self._mode = "logical"

    def set_legacy_collection(
        self,
        name: str,
        retriever_builder: "RetrieverBuilder",
        top_k: Optional[int] = None,
    ) -> None:
        irretriever = retriever_builder.build_for_legacy(name)
        self._activate_irretriever(irretriever, top_k or _resolve_top_k(retriever_builder))
        self.collection_name = name
        self._mode = "legacy_collection"

    def _activate_irretriever(self, irretriever: Any, top_k: int) -> None:
        adapter = LangChainRetrieverAdapter(irretriever=irretriever, top_k=top_k)
        self._build_chain_from_lc_retriever(adapter)

    def _build_chain_from_lc_retriever(self, retriever: Any) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(retriever, question_answer_chain)

    # ------------------------------------------------------------------
    # CLI loop (mantido)
    # ------------------------------------------------------------------

    def chat_loop(self) -> None:
        print("Iniciando Agente FII RAG (LangChain + Mistral v2 LCEL)...")
        try:
            self._setup_engine()
        except Exception as e:  # noqa: BLE001
            print(f"Erro ao iniciar a Engine. O Qdrant está rodando e persistido? Erro:\n{e}")
            return

        print("\n--- Agente Pronto! ---")
        print(
            "Faça perguntas sobre os relatórios inseridos. "
            "Digite 'sair' ou 'exit' para fechar."
        )

        while True:
            try:
                user_input = input("\n👤 Você: ")
                if user_input.lower() in ["sair", "exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                print("⏳ Pensando (Buscando + Reranking + LLM)...")
                response = self.chain.invoke({"input": user_input})
                answer = response.get("answer", "Não foi possível gerar resposta.")
                print(f"\n🤖 Mistral AI Responde:\n{answer}")
            except KeyboardInterrupt:
                break
            except Exception as e:  # noqa: BLE001
                print(f"Ocorreu um erro ao processar a resposta: {e}")


def _resolve_top_k(retriever_builder: "RetrieverBuilder") -> int:
    return retriever_builder.config.retrieval.final_top_k
