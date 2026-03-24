from src.generation.trie import FunctionNameTrie
from src.generation.json_types.type_registry import JSONTypeRegistry
from src.models import FunctionFormat
from llm_sdk import Small_LLM_Model
from enum import Enum, auto
import json


class DecodingSteps(Enum):
    START_TO_NAME = auto()
    GENERATE_NAME = auto()
    PARAMS_TO_FIRST_KEY = auto()
    KEY_PARAMS = auto()
    VALUE_PARAMS = auto()
    END = auto()


class JsonStateMachine:
    def __init__(
        self,
        model: Small_LLM_Model,
        prompt_text: str,
        functions: list[FunctionFormat],
        fn_name_trie: FunctionNameTrie,
        type_registry: JSONTypeRegistry,
    ):
        self.model = model
        self.escaped_prompt = json.dumps(prompt_text)
        self.functions = functions
        self.fn_name_trie = fn_name_trie
        self.type_registry = type_registry

        self.current_step = DecodingSteps.START_TO_NAME
        self.current_trie_node = self.fn_name_trie.root
        self.function_name: str | None = None
        self.function_params = []
        self.current_param_type: str | None = None
        self.is_first_param = True
        self.param_token_count = 0

    def can_fast_forward(self):
        if self.current_step in [
            DecodingSteps.START_TO_NAME,
            DecodingSteps.PARAMS_TO_FIRST_KEY,
            DecodingSteps.KEY_PARAMS,
            DecodingSteps.END,
        ]:
            return True
        return False

    def get_ff_tokens(self) -> list[int]:
        match self.current_step:
            case DecodingSteps.START_TO_NAME:
                self.current_step = DecodingSteps.GENERATE_NAME
                return self.model.encode(
                    f'{{\n  "prompt": {self.escaped_prompt},\n  "name": "'
                ).tolist()[0]

            case DecodingSteps.PARAMS_TO_FIRST_KEY:
                self.current_step = DecodingSteps.KEY_PARAMS
                return self.model.encode('",\n  "parameters": {').tolist()[0]

            case DecodingSteps.KEY_PARAMS:
                close_prev = '"' if self.current_param_type == "string" else ""
                if not self.function_params:
                    self.current_step = DecodingSteps.END
                    return self.model.encode(
                        f"{close_prev}\n  }}\n}}"
                    ).tolist()[0]
                param_name, param_info = self.function_params.pop(0)
                self.current_param_type = param_info.type
                prefix = (
                    f"{close_prev}\n    "
                    if self.is_first_param
                    else f"{close_prev},\n    "
                )
                self.is_first_param = False
                quote = '"' if self.current_param_type == "string" else ""
                self.current_step = DecodingSteps.VALUE_PARAMS
                self.param_token_count = 0
                return self.model.encode(
                    f'{prefix}"{param_name}": {quote}'
                ).tolist()[0]
            case _:
                return []

    def get_allowed_tokens(self) -> list[int]:
        match self.current_step:
            case DecodingSteps.GENERATE_NAME:
                return self.fn_name_trie.get_allowed_tokens(
                    self.current_trie_node
                )
            case DecodingSteps.VALUE_PARAMS:
                allowed_tokens: set[int] = (
                    self.type_registry.get_allowed_tokens_for_type(
                        self.current_param_type
                    )
                )
                if self.current_param_type == "string":
                    allowed_tokens.update(self.type_registry.string_end_tokens)
                else:
                    allowed_tokens.update(self.type_registry.number_end_tokens)
                return list(allowed_tokens)

            case _:
                return []

    def advance(self, next_token_id: int) -> list[int]:
        match self.current_step:
            case DecodingSteps.GENERATE_NAME:
                self.current_trie_node = self.current_trie_node.children[
                    next_token_id
                ]
                if self.current_trie_node.is_end:
                    self.function_name = self.current_trie_node.function_name
                    func_spec = next(
                        f
                        for f in self.functions
                        if f.name == self.function_name
                    )
                    self.function_params = list(func_spec.parameters.items())
                    self.current_step = DecodingSteps.PARAMS_TO_FIRST_KEY
                return [next_token_id]

            case DecodingSteps.VALUE_PARAMS:
                self.param_token_count += 1
                if self.current_param_type == "string":
                    if next_token_id in self.type_registry.string_end_tokens:
                        self.current_step = DecodingSteps.KEY_PARAMS
                        return self.type_registry.token_splits.get(
                            next_token_id, []
                        )

                    return [next_token_id]

                else:
                    if next_token_id in self.type_registry.number_end_tokens:
                        self.current_step = DecodingSteps.KEY_PARAMS
                        return self.type_registry.token_splits.get(
                            next_token_id, []
                        )

                    return [next_token_id]

            case _:
                return [next_token_id]

    def is_done(self):
        return self.current_step is DecodingSteps.END
