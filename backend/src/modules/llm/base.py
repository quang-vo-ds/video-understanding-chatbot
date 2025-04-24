import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable

from configs import settings

logger = logging.getLogger(settings.LOGGER)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, inp: str) -> str:
        pass

    @abstractmethod
    def stream(self, inp: str) -> Iterable:
        pass


class LLMFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: BaseLLM) -> Callable:
            if name in cls.registry:
                logger.warning(f"Name {name} already exists. Will replace it")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str) -> BaseLLM:
        if name not in cls.registry:
            logger.warning(f"Name {name} does not exist in the registry")
            return None

        return cls.registry[name]
