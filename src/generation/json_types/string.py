class JSONString:
    def __init__(self, vocab: dict[str, int]):
        self.preset_tokens = self._build_preset(vocab)

    def _build_preset(self, vocab: dict[str, int]) -> set[int]:
        return set(vocab.values())

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens
