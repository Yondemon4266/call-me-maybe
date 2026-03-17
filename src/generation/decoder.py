import json

from llm_sdk import Small_LLM_Model
from src.generation.decoding_steps import DecodingStep
from src.generation.state_machine import JsonStateMachine
from src.generation.trie import FunctionNameTrie
from src.models import FunctionFormat, PromptFormat


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
        self.functions_text = self._init_functions_text()
        self.context = (
            "You are a helpful assistant. Choose the correct "
            "function from this list and get the needed arguments "
            "based on the user prompt for it to work:\n\n"
            f"{self.functions_text}"
        )
        self.vocabulary_file = model.get_path_to_vocab_file()
        self.fn_name_trie = FunctionNameTrie(self.model, self.functions)

    def _init_functions_text(self) -> str:
        functions_text = ""
        for func in self.functions:
            functions_text += f"- Function: {func.name}\n"
            functions_text += f"  Description: {func.description}\n"
            functions_text += f"  Parameters: {func.parameters}\n"
            functions_text += f"  Returns: {func.returns}\n\n"
        return functions_text

    def _mask_logits(
        self, logits: list[float], allowed_ids: list[int]
    ) -> None:
        if not allowed_ids:
            return

        allowed_set = set(allowed_ids)
        for i in range(len(logits)):
            if i not in allowed_set:
                logits[i] = -float("inf")

    def generate_json_in_output_format(self, prompt: PromptFormat) -> dict:
        dynamic_context = (
            self.context + f"User request: '{prompt.prompt}'\n"
            "Now, generate the exact JSON to call the right function.\n\n"
        )
        input_ids: list[int] = self.model.encode(dynamic_context).tolist()[0]

        max_tokens = 200
        generated_tokens = []

        print(f"\n--- [PURISTE] Traitement de : {prompt.prompt} ---")

        # 1. On instancie notre arbitre pour cette génération précise
        fsm = JsonStateMachine(
            model=self.model,
            prompt_text=prompt.prompt,
            functions=self.functions,
            fn_name_trie=self.fn_name_trie,
        )

        for _ in range(max_tokens):

            # --- LE RACCOURCI : FAST-FORWARDING ---
            if fsm.current_step == DecodingStep.STEP_PROMPT_VALUE:
                prompt_tokens = fsm.fast_forward_prompt()
                input_ids.extend(prompt_tokens)
                generated_tokens.extend(prompt_tokens)
                print(prompt.prompt, end="", flush=True)
                continue
            # --------------------------------------

            logits = self.model.get_logits_from_input_ids(input_ids)

            # 2. On demande les règles à l'arbitre
            allowed_token_ids = fsm.get_allowed_tokens()

            # Sécurité temporaire : on arrête le code ici tant qu'on n'a pas codé la logique des clés !
            if fsm.current_step == DecodingStep.AI_PARAMS_KEYS:
                break

            # 3. Masquage et sélection
            self._mask_logits(logits, allowed_token_ids)
            next_token_id = logits.index(max(logits))

            input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)

            print(self.model.decode([next_token_id]), end="", flush=True)

            # 4. On informe l'arbitre du choix de l'IA. S'il dit stop, on coupe !
            is_finished = fsm.advance(next_token_id)
            if is_finished:
                break

        return {"raw": self.model.decode(generated_tokens)}
