from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TokenCounter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def count(self, text: str) -> int:
        ...


class TiktokenCounter(TokenCounter):
    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name

    @property
    def name(self) -> str:
        return f"tiktoken_{self._encoding_name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if not isinstance(text, str):
            text = str(text)
        return len(self._enc.encode(text))


class CharFallbackCounter(TokenCounter):
    @property
    def name(self) -> str:
        return "char_fallback"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class EB5SentencePieceCounter(TokenCounter):
    """Placeholder for EB5 SentencePiece tokenizer.
    Implement when model file is available from the team."""

    def __init__(self, model_path: str | Path):
        raise NotImplementedError(
            "EB5 SentencePiece tokenizer not yet available. "
            "Please provide the model file path once the team releases it."
        )

    @property
    def name(self) -> str:
        return "eb5_sentencepiece"

    def count(self, text: str) -> int:
        raise NotImplementedError


def create_tokenizer(name: str = "tiktoken") -> TokenCounter:
    if name == "tiktoken":
        try:
            return TiktokenCounter()
        except ImportError:
            return CharFallbackCounter()
    if name == "eb5":
        raise NotImplementedError(
            "EB5 tokenizer requires a model file. Use --tokenizer tiktoken or char for MVP."
        )
    return CharFallbackCounter()
