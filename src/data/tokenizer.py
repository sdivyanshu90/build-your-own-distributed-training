"""Thin HuggingFace tokenizer wrapper.

What this module does
---------------------
Wraps a ``transformers`` tokenizer behind a tiny, stable interface
(``encode``/``decode``/``vocab_size``) so the rest of the system never imports
``transformers`` directly. This keeps the dependency at the edge and lets tests
substitute a trivial byte tokenizer without pulling the library.

Why a wrapper
-------------
Different tokenizers expose slightly different APIs and special-token handling.
Pinning a narrow interface here means a tokenizer swap (GPT-2 BPE -> a custom
SentencePiece model) touches one file, and ``vocab_size`` always agrees with the
model's embedding dimension — a mismatch there is a silent source of index errors.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Structural interface the training pipeline relies on."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class HFTokenizer:
    """Adapter over a ``transformers`` fast tokenizer.

    Args:
        name_or_path: HF hub id or local path (e.g. ``"gpt2"``).

    Raises:
        ImportError: If ``transformers`` is not installed.

    Example:
        >>> tok = HFTokenizer("gpt2")           # doctest: +SKIP
        >>> ids = tok.encode("hello world")     # doctest: +SKIP
        >>> tok.decode(ids)                     # doctest: +SKIP
        'hello world'
    """

    def __init__(self, name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HFTokenizer requires `transformers`; install it or use "
                "ByteTokenizer for tests."
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(name_or_path)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    @property
    def vocab_size(self) -> int:
        return int(self._tok.vocab_size)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text))

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids))


class ByteTokenizer:
    """Dependency-free byte-level tokenizer (vocab=256) for tests/offline use."""

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def build_tokenizer(name_or_path: str | None) -> Tokenizer:
    """Return an :class:`HFTokenizer`, or :class:`ByteTokenizer` if name is falsy."""
    if not name_or_path:
        return ByteTokenizer()
    return HFTokenizer(name_or_path)
