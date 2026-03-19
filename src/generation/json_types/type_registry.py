import json
from src.generation.json_types.bool import JSONBool
from src.generation.json_types.integer import JSONInteger
from src.generation.json_types.number import JSONNumber
from src.generation.json_types.string import JSONString


class JSONTypeRegistry:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.bool_handler = JSONBool(self.vocab)
        self.int_handler = JSONInteger(self.vocab)
        self.number_handler = JSONNumber(self.vocab)
        self.string_handler = JSONString(self.vocab)
        self.comma_token = self.vocab.get(",")
        self.closing_brace_token = self.vocab.get("}")
        self.opening_brace_token = self.vocab.get("{")
        self.colon_token = self.vocab.get(":")
        self.quote_token = self.vocab.get('"')
        self.quote_comma_token = self.vocab.get('",')
        self.quote_brace_token = self.vocab.get('"}')

    def get_allowed_tokens_for_type(self, json_type: str) -> set[int]:
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
