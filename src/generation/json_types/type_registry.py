"""Registry for mapping JSON types to allowed tokenizer tokens."""

import json
from src.generation.json_types.json_types import (
    JSONBool,
    JSONInteger,
    JSONString,
    JSONNumber,
)
import sys
from llm_sdk import Small_LLM_Model
import re


class JSONTypeRegistry:
    """Manage JSON type handlers and token-level end-of-value rules."""

    def __init__(self, model: Small_LLM_Model, vocab_path: str):
        """Load vocabulary and initialize type-specific token filters.

        Args:
            model: Language model wrapper used for token encode/decode.
            vocab_path: Filesystem path to the tokenizer vocabulary JSON.

        Returns:
            None.

        Raises:
            SystemExit: If the vocabulary file cannot be read.
        """
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab: dict[str, int] = json.load(f)
        except OSError as e:
            sys.stderr.write(
                f"Error while trying to read AI vocabulary file {e.filename}"
                "file may be corrupted or non-existant"
            )
            sys.exit(1)

        self.model = model
        self.bool_handler = JSONBool(self.model, self.vocab)
        self.int_handler = JSONInteger(self.model, self.vocab)
        self.number_handler = JSONNumber(self.model, self.vocab)
        self.string_handler = JSONString(self.model, self.vocab)
        self.string_end_tokens: set[int] = set()
        self.number_end_tokens: set[int] = set()
        self.token_splits: dict[int, list[int]] = {}
        self._build_end_rules()

    def get_allowed_tokens_for_type(self, json_type: str | None) -> set[int]:
        """Return allowed token IDs for a given JSON scalar type.

        Args:
            json_type: One of supported scalar JSON types.

        Returns:
            Set of token IDs accepted for the requested type.
        """
        match json_type:
            case "boolean":
                return self.bool_handler.get_allowed_tokens()
            case "integer":
                return self.int_handler.get_allowed_tokens()
            case "number":
                return self.number_handler.get_allowed_tokens()
            case "string":
                return self.string_handler.get_allowed_tokens()
            case _:
                return set()

    def _build_end_rules(self) -> None:
        """Build rules used to end string and numeric JSON values.

        Returns:
            None.
        """
        num_pattern = re.compile(r"^\s*-?[0-9]*\.?[0-9]*$")

        for token_str, token_id in self.vocab.items():
            token_str = self.model.decode([token_id])
            match = re.search(r"[,}\n]", token_str)
            match_quote = re.search(r'(?<!\\)"', token_str)

            if match_quote:
                self.string_end_tokens.add(token_id)
                text_before = token_str[: match_quote.start()]
                self.token_splits[token_id] = (
                    self.model.encode(text_before).tolist()[0]
                    if text_before
                    else []
                )
            elif match:
                text_before = token_str[: match.start()]
                if not text_before or num_pattern.fullmatch(text_before):
                    self.number_end_tokens.add(token_id)
                    self.token_splits[token_id] = (
                        self.model.encode(text_before).tolist()[0]
                        if text_before
                        else []
                    )
