import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class AppConfig:
    """
    Classe responsável pela configuração do ecossistema principal.
    """
    
    def __init__(self):
        load_dotenv()
        self._mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self._google_api_key = os.getenv("GOOGLE_API_KEY")
        self._qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self._mistral_model = os.getenv("MISTRAL_MODEL", "codestral-embed-25-05")
        self._google_model = os.getenv("GOOGLE_MODEL", "gemini-embedding-2-preview")

    @property
    def qdrant_url(self) -> str:
        return self._qdrant_url

    def get_llm_and_embeddings(self):
        """
        Retorna as instâncias dos modelos LangChain.
        """
        if not self._mistral_api_key:
            print("Aviso: MISTRAL_API_KEY não encontrada no .env.")
            
        if not self._google_api_key:
            print("Aviso: GOOGLE_API_KEY não encontrada no .env.")

        llm = None
        embed_model = None

        if self._mistral_api_key:
            llm = ChatMistralAI(
                model=self._mistral_model,
                mistral_api_key=self._mistral_api_key,
                temperature=0.1
            )

        if self._google_api_key:
            embed_model = GoogleGenerativeAIEmbeddings(
                model=self._google_model,
                google_api_key=self._google_api_key,
            )
        
        return llm, embed_model
