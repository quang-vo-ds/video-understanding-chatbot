import pytest

from modules.llm import LLMFactory


def test_gen_str():
    base_llm = LLMFactory.create("Ollama")
    llm = base_llm(llm_name="llama3.1", base_url="http://localhost:11434")

    prompt = "Please translate 'I love programming.' to German"

    output = llm.generate(prompt)

    print(output)

    assert len(output) > 0


def test_stream():
    base_llm = LLMFactory.create("Ollama")
    llm = base_llm(llm_name="llama3.1", base_url="http://localhost:11434")

    prompt = "Tell me about yourself"

    output = llm.stream(prompt)

    assert next(output, None) is not None

    for token in output:
        print(token, end="", flush=True)
