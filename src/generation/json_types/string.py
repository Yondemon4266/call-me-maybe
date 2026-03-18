class JSONString:
    def __init__(self, vocab: dict[str, int]):
        self.preset_tokens = self._build_preset(vocab)

    def _build_preset(self, vocab: dict[str, int]) -> set[int]:
        valid_ids = set()

        for token_str, token_id in vocab.items():
            if '"' not in token_str:
                valid_ids.add(token_id)

        return valid_ids

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens
