import json
from llm_sdk import Small_LLM_Model
from src.generation.state_machine import JsonStateMachine, DecodingStep
from src.generation.trie import FunctionNameTrie
from src.generation.json_types.type_registry import JSONTypeRegistry
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
        self.vocabulary_file = model.get_path_to_vocab_file()

        # Initialisation des structures globales (Faites UNE seule fois)
        self.fn_name_trie = FunctionNameTrie(self.model, self.functions)
        self.type_registry = JSONTypeRegistry(self.vocabulary_file)

        # On construit le contexte complet pour l'IA
        self.context = (
            "You are a helpful assistant. Choose the correct "
            "function from this list and get the needed arguments "
            "based on the user prompt for it to work. "
            "Keep all extracted parameters "
            "as short and concise as possible. "
            "Do not extract full sentences if a short pattern is sufficient."
            "CRITICAL INSTRUCTIONS FOR REGEX:\n"
            "- Always use the absolute SHORTEST regex pattern possible (e.g., '\\d+', '[a-z]+').\n"
            "- NEVER capture full sentences. NEVER repeat capture groups.\n\n"
        )
        for func in self.functions:
            self.context += f"- Function: {func.name}\n"
            self.context += f"  Description: {func.description}\n"
            self.context += f"  Parameters: {func.parameters}\n"
            self.context += f"  Returns: {func.returns}\n\n"

    def _mask_logits(
        self, logits: list[float], allowed_ids: list[int] | set[int]
    ) -> None:
        if not allowed_ids:
            return
        # Set pour une recherche instantanée (O(1))
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

        generated_tokens = []

        # 1. On lance l'arbitre pour cette génération
        fsm = JsonStateMachine(
            model=self.model,
            prompt_text=prompt.prompt,
            functions=self.functions,
            fn_name_trie=self.fn_name_trie,
            type_registry=self.type_registry,
        )

        # 2. La boucle principale ultra-épurée
        while not fsm.is_done():

            # A. L'arbitre veut-il sauter des étapes (Fast-Forward) ?
            if fsm.can_fast_forward():
                ff_tokens = fsm.get_fast_forward_tokens()
                input_ids.extend(ff_tokens)
                generated_tokens.extend(ff_tokens)
                print(self.model.decode(ff_tokens), end="", flush=True)
                continue

            # B. Génération normale de l'IA
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_tokens = fsm.get_allowed_tokens()

            self._mask_logits(logits, allowed_tokens)
            next_token_id = logits.index(max(logits))

            input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)
            print(self.model.decode([next_token_id]), end="", flush=True)

            # C. On notifie l'arbitre du choix
            fsm.advance(next_token_id)

        return {"raw": self.model.decode(generated_tokens)}
