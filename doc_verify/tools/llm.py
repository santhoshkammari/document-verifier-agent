    # LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama


class LLM:
    def call_model(self, *args, **kwargs):
        llm = ChatOllama(model="llama3",temperature=0,verbose=True)
        return llm


