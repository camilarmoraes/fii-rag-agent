from server.src.fii_rag.interfaces import IQueryEngineBuilder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import Any

class RAGAgent:
    """
    Controlador do chat utilizando a engine LangChain (LCEL).
    """
    def __init__(self, query_engine_builder: IQueryEngineBuilder, llm: Any):
        self.query_engine_builder = query_engine_builder
        self.llm = llm
        self.chain = None
        self.collection_name = "fii_reports"

    def _setup_engine(self):
        if not self.chain:
            # Puxa o Retriever (CrossEncoder Baseado)
            retriever = self.query_engine_builder.build(self.collection_name)
            
            # Monta o prompt de sistema especializado RAG
            system_prompt = (
                "Você é um assistente sênior especializado em Fundos Imobiliários Brasileiros (FIIs). "
                "Use os seguintes trechos de contexto recuperado para responder a pergunta. "
                "Se você não souber a resposta, seja honesto e diga que não sabe. "
                "Responda de forma clara e estruturada.\n\n"
                "Aqui está o contexto extraído dos relatórios:\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # LCEL: Chain que junta Documentos no Contexto do Prompt
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # LCEL: Chain final que junta a Busca no Retriever com o Answering
            self.chain = create_retrieval_chain(retriever, question_answer_chain)

    def chat_loop(self):
        print("Iniciando Agente FII RAG (LangChain + Mistral v2 LCEL)...")
        
        try:
            self._setup_engine()
        except Exception as e:
            print(f"Erro ao iniciar a Engine. O Qdrant está rodando e persistido? Erro:\n{e}")
            return

        print("\n--- Agente Pronto! ---")
        print("Faça perguntas sobre os relatórios inseridos. Digite 'sair' ou 'exit' para fechar.")
        
        while True:
            try:
                user_input = input("\n👤 Você: ")
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break
                    
                if not user_input.strip():
                    continue
                
                print("⏳ Pensando (Buscando Hybrid + Reranking + Mistral 2.x)...")
                
                # Executa o Runnable do LangChain
                response = self.chain.invoke({"input": user_input})
                answer = response.get("answer", "Não foi possível gerar resposta.")
                
                print(f"\n🤖 Mistral AI Responde:\n{answer}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ocorreu um erro ao processar a resposta do Mistral: {e}")
