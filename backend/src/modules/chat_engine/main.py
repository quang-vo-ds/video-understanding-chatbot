import logging

from configs import settings
from modules.llm import BaseLLM

from . import template


class ChatEngine:
    def __init__(
        self, llm: BaseLLM, logger: logging = logging.getLogger(settings.LOGGER)
    ) -> None:
        super().__init__()
        self.llm = llm
        self.logger = logger

    def invoke(self, query: str, context: str) -> str:
        prompt = template.DEFAULT.format(question=query, context=context)

        res = self.llm.generate(prompt)

        self.logger.info(f"Conversation prompt:\n{prompt}")
        self.logger.info(f"Conversation result:\n{res}")

        return res
