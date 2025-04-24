from collections.abc import Iterable

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from configs import settings
from modules.llm.base import BaseLLM, LLMFactory


@LLMFactory.register("Ollama")
class OLlama(BaseLLM):
    def __init__(
        self,
        llm_name: str = settings.LLM_NAME,
        base_url: str = settings.LLM_URL,
        key: str = "",
        temprature: float = 0.2,
        request_timeout: float = 120,
    ) -> None:
        super().__init__()
        self.llm = ChatOllama(
            base_url=base_url,
            model=llm_name,
            request_timeout=request_timeout,
            temperature=temprature,
        )

    def generate(self, prompt: str) -> str:
        engine = self.llm | StrOutputParser()
        return engine.invoke(prompt)

    def stream(self, prompt: str, eos: str = '<EOS>') -> Iterable:
        engine = self.llm | StrOutputParser()
        for chunk in engine.stream(prompt):
            yield chunk
        yield eos
