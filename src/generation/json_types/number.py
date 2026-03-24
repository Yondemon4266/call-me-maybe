import re
from llm_sdk import Small_LLM_Model


class JSONNumber:
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        self.model = model
        self.preset_tokens = self._build_preset(vocab)

    def _build_preset(self, vocab: dict[str, int]) -> set[int]:
        valid_ids: set[int] = set()
        pattern = re.compile(r"^\s*-?[0-9]*\.?[0-9]*$")

        for token_id in vocab.values():
            token_str = self.model.decode([token_id])
            # Plus de 'or any(...)', juste la regex stricte !
            if pattern.fullmatch(token_str):
                valid_ids.add(token_id)
        return valid_ids

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens
