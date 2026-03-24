import re
from llm_sdk import Small_LLM_Model


class JSONInteger:
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        self.preset_tokens = self._build_preset(model, vocab)

    def _build_preset(self, model, vocab) -> set[int]:
        valid_ids = set()
        pattern = re.compile(r"^ * -?[0-9]*$")

        for token_id in vocab.values():
            token_str = model.decode([token_id])
            if pattern.fullmatch(token_str) or any(
                c in token_str for c in [",", "}", "\n"]
            ):
                valid_ids.add(token_id)
        return valid_ids

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens
