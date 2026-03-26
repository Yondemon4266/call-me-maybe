"""Token validators for supported JSON scalar types."""

import re
from llm_sdk import Small_LLM_Model


class JSONBaseType:
    """Base token filter for JSON scalar value generation."""

    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: dict[str, int],
        regex_pattern: str | None = None,
    ):
        """Create a token filter from an optional validation regex.

        Args:
            model: Language model wrapper used for token decoding.
            vocab: Vocabulary mapping token strings to token IDs.
            regex_pattern: Optional full-match pattern for valid tokens.

        Returns:
            None.
        """
        self.model = model
        self.pattern = re.compile(regex_pattern) if regex_pattern else None
        self.preset_tokens = self._build_preset(vocab)

    def _build_preset(self, vocab: dict[str, int]) -> set[int]:
        """Precompute all token IDs that satisfy the validator.

        Args:
            vocab: Vocabulary mapping token strings to token IDs.

        Returns:
            Set of valid token IDs.
        """
        valid_ids = set()
        for token_id in vocab.values():
            token_str = self.model.decode([token_id])
            if self._is_valid_token(token_str):
                valid_ids.add(token_id)
        return valid_ids

    def _is_valid_token(self, token_str: str) -> bool:
        """Check whether a decoded token is valid for this JSON type.

        Args:
            token_str: Decoded token string.

        Returns:
            True when token matches the configured constraints.
        """
        if self.pattern:
            return bool(self.pattern.fullmatch(token_str))
        return True

    def get_allowed_tokens(self) -> set[int]:
        """Return precomputed valid token IDs.

        Returns:
            Set of token IDs allowed for this type.
        """
        return self.preset_tokens


class JSONBool(JSONBaseType):
    """Token filter for JSON boolean values."""

    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        """Initialize boolean token validator.

        Args:
            model: Language model wrapper used for token decoding.
            vocab: Vocabulary mapping token strings to token IDs.

        Returns:
            None.
        """
        super().__init__(model, vocab, r"^\s*(true|false)$")


class JSONInteger(JSONBaseType):
    """Token filter for JSON integer values."""

    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        """Initialize integer token validator.

        Args:
            model: Language model wrapper used for token decoding.
            vocab: Vocabulary mapping token strings to token IDs.

        Returns:
            None.
        """
        super().__init__(model, vocab, r"^\s*-?[0-9]*$")


class JSONNumber(JSONBaseType):
    """Token filter for JSON number values."""

    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        """Initialize floating-point token validator.

        Args:
            model: Language model wrapper used for token decoding.
            vocab: Vocabulary mapping token strings to token IDs.

        Returns:
            None.
        """
        super().__init__(model, vocab, r"^\s*-?[0-9]*\.?[0-9]*$")


class JSONString(JSONBaseType):
    """Token filter for JSON string content values."""

    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        """Initialize string token validator.

        Args:
            model: Language model wrapper used for token decoding.
            vocab: Vocabulary mapping token strings to token IDs.

        Returns:
            None.
        """
        super().__init__(model, vocab)

    def _is_valid_token(self, token_str: str) -> bool:
        """Disallow tokens containing newline characters.

        Args:
            token_str: Decoded token string.

        Returns:
            True when token does not contain line breaks.
        """
        return "\n" not in token_str
