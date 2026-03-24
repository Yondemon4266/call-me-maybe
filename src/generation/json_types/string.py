from llm_sdk import Small_LLM_Model

class JSONString:
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]):
        self.preset_tokens = self._build_preset(model, vocab)

    def _build_preset(self, model: Small_LLM_Model, vocab: dict[str, int]) -> set[int]:
        valid_ids = set()
        
        for token_id in vocab.values():
            token_str = model.decode([token_id])
            # La norme JSON interdit les sauts de ligne réels dans les strings
            if "\n" not in token_str:
                valid_ids.add(token_id)
                
        return valid_ids

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens