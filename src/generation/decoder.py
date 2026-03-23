from llm_sdk import Small_LLM_Model
from src.generation.state_machine import JsonStateMachine
from src.generation.trie import FunctionNameTrie
from src.generation.json_types.type_registry import JSONTypeRegistry
from src.models import FunctionFormat, PromptFormat
import numpy as np


class JsonConstrainedDecoder:
    def __init__(
        self,
        model: Small_LLM_Model,
        prompts: list[PromptFormat],
        functions: list[FunctionFormat],
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.functions = functions
        self.vocabulary_file = model.get_path_to_vocab_file()

        self.fn_name_trie = FunctionNameTrie(self.model, self.functions)
        self.type_registry = JSONTypeRegistry(self.model, self.vocabulary_file)

        self.context = (
            "SYSTEM: Extract arguments as a JSON tool.\n"
            "RULES: No chat. No math. Exact extraction only.\n"
            "FUNCTIONS:\n"
        )
        for func in self.functions:
            self.context += f"- {func.name}({func.parameters})\n"

    def generate_one_prompt_in_json(self, prompt: PromptFormat) -> list[int]:
        dynamic_context = (
            self.context + f"INPUT: '{prompt.prompt}'\n" "JSON_OUTPUT:\n"
        )
        input_ids: list[int] = self.model.encode(dynamic_context).tolist()[0]

        generated_tokens: list[int] = []

        fsm = JsonStateMachine(
            model=self.model,
            prompt_text=prompt.prompt,
            functions=self.functions,
            fn_name_trie=self.fn_name_trie,
            type_registry=self.type_registry,
        )

        while not fsm.is_done():
            if fsm.can_fast_forward():
                ff_tokens = fsm.get_ff_tokens()
                input_ids.extend(ff_tokens)
                generated_tokens.extend(ff_tokens)
                print(self.model.decode(ff_tokens), end="", flush=True)
                continue

            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_tokens: list[int] = fsm.get_allowed_tokens()

            logits_array = np.array(logits)
            if allowed_tokens:
                mask = np.ones(len(logits_array), dtype=bool)
                mask[allowed_tokens] = False
                logits_array[mask] = -np.inf

            next_token_id = int(np.argmax(logits_array))
            keep_token = fsm.advance(next_token_id)
            if keep_token:
                input_ids.append(next_token_id)
                generated_tokens.append(next_token_id)
                print(self.model.decode([next_token_id]), end="", flush=True)

        return generated_tokens

    def generate_all_prompts_in_json(self):
        all_generated_tokens: list[int] = [
            self.type_registry.open_bracket_token
        ]
        for prompt in self.prompts:
            all_generated_tokens.extend(
                self.generate_one_prompt_in_json(prompt)
            )
            all_generated_tokens.append(self.type_registry.comma_token)
        all_generated_tokens.append(self.type_registry.close_bracket_token)
