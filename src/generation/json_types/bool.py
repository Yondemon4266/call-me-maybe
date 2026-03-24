import re
from llm_sdk import Small_LLM_Model


class JSONBool:
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        self.preset_tokens = self._build_preset(model, vocab)

    def _build_preset(
        self, model: Small_LLM_Model, vocab: dict[str, int]
    ) -> set[int]:
        valid_ids: set[int] = set()
        pattern = re.compile(r"^\s*(true|false)$")

        for token_id in vocab.values():
            token_str = model.decode([token_id])
            if pattern.fullmatch(token_str):
                valid_ids.add(token_id)

        return valid_ids

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens
