from src.generation.trie import FunctionNameTrie
from src.generation.json_types.type_registry import JSONTypeRegistry
from src.models import FunctionFormat
from llm_sdk import Small_LLM_Model
from enum import Enum, auto


class DecodingStep(Enum):
    PREFIX_TO_NAME = auto()
    GENERATE_NAME = auto()
    PREFIX_TO_PARAMS = auto()
    GENERATE_PARAM_KEY = auto()
    GENERATE_PARAM_VALUE = auto()
    WAIT_FOR_COMMA_OR_BRACE = auto()
    CLOSE_JSON = auto()  # --- NOUVEAU : État pour fermer le fichier ---
    DONE = auto()


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
        self.functions = functions
        self.fn_name_trie = fn_name_trie
        self.type_registry = type_registry

        self.prefix_1_text = f'{{\n  "prompt": "{prompt_text}",\n  "name": "'
        self.prefix_1_tokens = self.model.encode(self.prefix_1_text).tolist()[
            0
        ]

        self.prefix_2_text = '",\n  "parameters": {'
        self.prefix_2_tokens = self.model.encode(self.prefix_2_text).tolist()[
            0
        ]

        # --- NOUVEAU : On récupère le Token ID du saut de ligne ---
        self.newline_token = self.model.encode("\n").tolist()[0][0]
        self.closing_token_used = None

        self.current_step = DecodingStep.PREFIX_TO_NAME
        self.current_trie_node = self.fn_name_trie.root

        self.chosen_function: str | None = None
        self.remaining_params = []
        self.current_param_name: str | None = None
        self.current_param_type: str | None = None

    def can_fast_forward(self) -> bool:
        return self.current_step in [
            DecodingStep.PREFIX_TO_NAME,
            DecodingStep.PREFIX_TO_PARAMS,
            DecodingStep.GENERATE_PARAM_KEY,
            DecodingStep.CLOSE_JSON,  # --- NOUVEAU
        ]

    def get_fast_forward_tokens(self) -> list[int]:
        if self.current_step == DecodingStep.PREFIX_TO_NAME:
            self.current_step = DecodingStep.GENERATE_NAME
            return self.prefix_1_tokens

        elif self.current_step == DecodingStep.PREFIX_TO_PARAMS:
            self.current_step = DecodingStep.GENERATE_PARAM_KEY
            return self.prefix_2_tokens

        elif self.current_step == DecodingStep.GENERATE_PARAM_KEY:
            if self.remaining_params:
                param_name, param_info = self.remaining_params.pop(0)
                self.current_param_name = param_name
                self.current_param_type = param_info.type

                key_text = f'\n    "{param_name}": '
                if self.current_param_type == "string":
                    key_text += '"'

                self.current_step = DecodingStep.GENERATE_PARAM_VALUE
                return self.model.encode(key_text).tolist()[0]

        # --- NOUVEAU : Fast-forward de la fermeture du JSON ---
        elif self.current_step == DecodingStep.CLOSE_JSON:
            self.current_step = DecodingStep.DONE
            if self.closing_token_used == self.newline_token:
                # Si l'IA a fait "Entrée", on rajoute les espaces et les accolades
                return self.model.encode("  }\n}").tolist()[0]
            else:
                # Si l'IA a écrit "}", on ferme juste la fin
                return self.model.encode("\n}").tolist()[0]

        return []

    def get_allowed_tokens(self) -> list[int] | set[int]:
        if self.current_step == DecodingStep.GENERATE_NAME:
            return self.fn_name_trie.get_allowed_tokens(self.current_trie_node)

        elif self.current_step == DecodingStep.GENERATE_PARAM_VALUE:
            allowed = self.type_registry.get_allowed_tokens_for_type(
                self.current_param_type
            ).copy()

            if self.current_param_type == "string":
                allowed.add(self.type_registry.quote_token)
                allowed.add(self.type_registry.quote_brace_token)
                allowed.add(self.type_registry.quote_comma_token)
            else:
                if self.remaining_params:
                    allowed.add(self.type_registry.comma_token)
                else:
                    allowed.add(self.type_registry.closing_brace_token)
                    allowed.add(
                        self.newline_token
                    )  # --- NOUVEAU : Autoriser l'IA à passer à la ligne !
            return allowed

        elif self.current_step == DecodingStep.WAIT_FOR_COMMA_OR_BRACE:
            if self.remaining_params:
                return [self.type_registry.comma_token]
            else:
                return [
                    self.type_registry.closing_brace_token,
                    self.newline_token,
                ]

        return []

    def advance(self, token_id: int):
        if self.current_step == DecodingStep.GENERATE_NAME:
            self.current_trie_node = self.current_trie_node.children[token_id]

            if self.current_trie_node.is_end:
                self.chosen_function = self.current_trie_node.function_name
                self.current_step = DecodingStep.PREFIX_TO_PARAMS
                func_spec = next(
                    f for f in self.functions if f.name == self.chosen_function
                )
                self.remaining_params = list(func_spec.parameters.items())

        elif self.current_step == DecodingStep.GENERATE_PARAM_VALUE:
            if self.current_param_type == "string":
                if token_id == self.type_registry.quote_token:
                    self.current_step = DecodingStep.WAIT_FOR_COMMA_OR_BRACE
            else:
                if self.remaining_params:
                    if token_id == self.type_registry.comma_token:
                        self.current_step = DecodingStep.GENERATE_PARAM_KEY
                else:
                    # --- NOUVEAU : Intercepter la fin ---
                    if token_id in [
                        self.type_registry.closing_brace_token,
                        self.newline_token,
                    ]:
                        self.closing_token_used = token_id
                        self.current_step = DecodingStep.CLOSE_JSON

        elif self.current_step == DecodingStep.WAIT_FOR_COMMA_OR_BRACE:
            if self.remaining_params:
                if token_id == self.type_registry.comma_token:
                    self.current_step = DecodingStep.GENERATE_PARAM_KEY
            else:
                # --- NOUVEAU : Intercepter la fin ---
                if token_id in [
                    self.type_registry.closing_brace_token,
                    self.newline_token,
                ]:
                    self.closing_token_used = token_id
                    self.current_step = DecodingStep.CLOSE_JSON

    def is_done(self) -> bool:
        return self.current_step == DecodingStep.DONE
