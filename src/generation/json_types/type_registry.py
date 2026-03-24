import json
from src.generation.json_types.bool import JSONBool
from src.generation.json_types.integer import JSONInteger
from src.generation.json_types.number import JSONNumber
from src.generation.json_types.string import JSONString
import sys
from llm_sdk import Small_LLM_Model
import re


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
        self.bool_handler = JSONBool(self.model, self.vocab)
        self.int_handler = JSONInteger(self.model, self.vocab)
        self.number_handler = JSONNumber(self.model, self.vocab)
        self.string_handler = JSONString(self.vocab)
        self.string_end_tokens = set()
        self.number_end_tokens = set()
        self.token_splits: dict[int, list[int]] = {}
        self._build_end_rules()

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

    def _build_end_rules(self) -> None:
        num_pattern = re.compile(r"^\s*-?[0-9]*\.?[0-9]*$")
        
        for token_str, token_id in self.vocab.items():
            token_str = self.model.decode([token_id])
            match = re.search(r"[,}\n]", token_str)
            match_quote = re.search(r'(?<!\\)"', token_str)
            
            if match_quote:
                self.string_end_tokens.add(token_id)
                text_before = token_str[: match_quote.start()]
                self.token_splits[token_id] = (
                    self.model.encode(text_before).tolist()[0] if text_before else []
                )
            elif match:
                text_before = token_str[: match.start()]
                if not text_before or num_pattern.fullmatch(text_before):
                    self.number_end_tokens.add(token_id)
                    self.token_splits[token_id] = (
                        self.model.encode(text_before).tolist()[0] if text_before else []
                    )