import re
from llm_sdk import Small_LLM_Model


class JSONBaseType:
    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: dict[str, int],
        regex_pattern: str | None = None,
    ):
        self.model = model
        self.pattern = re.compile(regex_pattern) if regex_pattern else None
        self.preset_tokens = self._build_preset(vocab)

    def _build_preset(self, vocab: dict[str, int]) -> set[int]:
        valid_ids = set()
        for token_id in vocab.values():
            token_str = self.model.decode([token_id])
            if self._is_valid_token(token_str):
                valid_ids.add(token_id)
        return valid_ids

    def _is_valid_token(self, token_str: str) -> bool:
        if self.pattern:
            return bool(self.pattern.fullmatch(token_str))
        return True

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens


class JSONBool(JSONBaseType):
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        super().__init__(model, vocab, r"^\s*(true|false)$")


class JSONInteger(JSONBaseType):
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        super().__init__(model, vocab, r"^\s*-?[0-9]*$")


class JSONNumber(JSONBaseType):
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        super().__init__(model, vocab, r"^\s*-?[0-9]*\.?[0-9]*$")


class JSONString(JSONBaseType):
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        super().__init__(model, vocab)

    def _is_valid_token(self, token_str: str) -> bool:
        return "\n" not in token_str
