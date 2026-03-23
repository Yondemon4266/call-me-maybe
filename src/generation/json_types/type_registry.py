import json
from src.generation.json_types.bool import JSONBool
from src.generation.json_types.integer import JSONInteger
from src.generation.json_types.number import JSONNumber
from src.generation.json_types.string import JSONString
import sys
from llm_sdk import Small_LLM_Model


class JSONTypeRegistry:
    def __init__(self, model: Small_LLM_Model, vocab_path: str):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab: dict[str, int] = json.load(f)
        except OSError as e:
            sys.stderr.write(
                f"Error while trying to read AI vocabulary file {e.filename}"
                "file may be corrupted or non-existant"
            )
            sys.exit(1)

        self.model = model
        self.bool_handler = JSONBool(self.vocab)
        self.int_handler = JSONInteger(self.vocab)
        self.number_handler = JSONNumber(self.vocab)
        self.string_handler = JSONString(self.vocab)
        self.open_bracket_token: int = self.model.encode("[").tolist()[0][0]
        self.close_bracket_token: int = self.model.encode("]").tolist()[0][0]
        self.comma_token: int = self.model.encode(",").tolist()[0][0]

    def get_allowed_tokens_for_type(self, json_type: str | None) -> set[int]:
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
