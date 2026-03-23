import re


class JSONNumber:
    def __init__(self, vocab: dict[str, int]):
        self.preset_tokens = self._build_preset(vocab)

    def _build_preset(self, vocab: dict[str, int]) -> set[int]:
        valid_ids: set[int] = set()
        pattern = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")

        termination_chars = [",", "}", "\n", ",\n", "}\n"]

        for token_str, token_id in vocab.items():
            if (
                pattern.match(token_str)
                or token_str.strip() in termination_chars
            ):
                valid_ids.add(token_id)

        return valid_ids

    def get_allowed_tokens(self) -> set[int]:
        return self.preset_tokens
