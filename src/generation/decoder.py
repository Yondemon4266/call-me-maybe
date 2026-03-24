from llm_sdk import Small_LLM_Model
from src.generation.state_machine import JsonStateMachine
from src.generation.trie import FunctionNameTrie
from src.generation.json_types.type_registry import JSONTypeRegistry
from src.models import FunctionFormat, PromptFormat
import numpy as np
import re
import json
import sys


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
            "Always preserve negative signs (e.g., -9999, -0).\n"
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
            # top_3 = np.argsort(logits_array)[-3:][::-1]
            # print("[DEBUG] Top 3 envies : " + " | ".join(f"{repr(self.model.decode([t]))} ({logits_array[t]:.2f})" for t in top_3 if logits_array[t] != -np.inf))
            next_token_id = int(np.argmax(logits_array))
            tokens_to_keep = fsm.advance(next_token_id)
            if tokens_to_keep:
                input_ids.extend(tokens_to_keep)
                generated_tokens.extend(tokens_to_keep)
                print(self.model.decode(tokens_to_keep), end="", flush=True)

        return generated_tokens

    def generate_all_prompts_in_json(self) -> str:
        prompts_list = []

        for prompt in self.prompts:
            prompt_tokens = self.generate_one_prompt_in_json(prompt)
            prompt_str = self.model.decode(prompt_tokens)
            cleaned_str = re.sub(
                r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", prompt_str
            )
            try:
                data = json.loads(cleaned_str)
                prompts_list.append(data)
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"Error: {prompt.prompt} does not contain valid JSON."
                    f"\nDetails: {e.msg} at line: {e.lineno}, col: {e.colno}",
                )
                continue
            except Exception as e:
                sys.stderr.write(
                    "\n[Erreur critique] Échec lors du traitement de "
                    f"'{prompt.prompt}'.\n"
                    f"Détails: {str(e)}\n"
                )
                continue
        return json.dumps(prompts_list, indent=2, ensure_ascii=False)
