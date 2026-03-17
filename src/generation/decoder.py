from collections import deque

from llm_sdk import Small_LLM_Model
from src.generation.fsm import DecodingStep
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

        self.static_steps_text = {
            DecodingStep.STEP_START_TO_PROMPT: '{\n  "prompt": "',
            DecodingStep.STEP_TO_NAME: '",\n  "name": "',
            DecodingStep.STEP_TO_PARAMS: '",\n  "parameters": {',
            DecodingStep.STEP_END: "\n}",
        }

        self.encoded_static_steps: dict[DecodingStep, list[int]] = {
            step: self.model.encode(text).tolist()[0]
            for step, text in self.static_steps_text.items()
        }
        self.fn_name_trie = FunctionNameTrie(self.model, self.functions)
        self.transitions = {
            DecodingStep.STEP_START_TO_PROMPT: DecodingStep.STEP_PROMPT_VALUE,
            DecodingStep.STEP_PROMPT_VALUE: DecodingStep.STEP_TO_NAME,
            DecodingStep.STEP_TO_NAME: DecodingStep.AI_NAME_VALUE,
            DecodingStep.AI_NAME_VALUE: DecodingStep.STEP_TO_PARAMS,
            DecodingStep.STEP_TO_PARAMS: DecodingStep.AI_PARAMS_KEYS,
            DecodingStep.AI_PARAMS_KEYS: DecodingStep.AI_PARAMS_VALS,
            DecodingStep.AI_PARAMS_VALS: DecodingStep.STEP_END,
        }

    def _init_functions_text(self) -> str:
        # On initialise une vraie chaîne de caractères vide (ou une liste qu'on .join() à la fin)
        functions_text = ""

        for func in self.functions:
            functions_text += f"- Function: {func.name}\n"
            functions_text += f"  Description: {func.description}\n"

            # On utilise json.dumps pour que l'IA lise un format clair et familier
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

        steps_map = self.encoded_static_steps.copy()
        steps_map[DecodingStep.STEP_PROMPT_VALUE] = self.model.encode(
            prompt.prompt
        ).tolist()[0]

        current_step = DecodingStep.STEP_START_TO_PROMPT
        current_step_queue = deque(steps_map[current_step])
        current_trie_node = self.fn_name_trie.root
        max_tokens = 200
        generated_tokens = []
        chosen_function = None

        current_params_queue = []
        current_param_type = None
        is_first_param = True
        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_token_ids = []

            match current_step:
                case (
                    DecodingStep.STEP_START_TO_PROMPT
                    | DecodingStep.STEP_PROMPT_VALUE
                    | DecodingStep.STEP_TO_NAME
                    | DecodingStep.STEP_TO_PARAMS
                    | DecodingStep.STEP_END
                ):
                    allowed_token_ids = [current_step_queue[0]]
                case DecodingStep.AI_NAME_VALUE:
                    allowed_token_ids = self.fn_name_trie.get_allowed_tokens(
                        current_trie_node
                    )
                case DecodingStep.AI_PARAMS_KEYS:
                    break
                case DecodingStep.AI_PARAMS_VALS:
                    pass

            self._mask_logits(logits, allowed_token_ids)

            next_token_id = logits.index(max(logits))
            input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)

            print(self.model.decode([next_token_id]), end="", flush=True)

            match current_step:
                case (
                    DecodingStep.STEP_START_TO_PROMPT
                    | DecodingStep.STEP_PROMPT_VALUE
                    | DecodingStep.STEP_TO_NAME
                    | DecodingStep.STEP_TO_PARAMS
                ):
                    current_step_queue.popleft()
                    if not current_step_queue:
                        current_step = self.transitions[current_step]
                        if current_step in steps_map:
                            current_step_queue = deque(steps_map[current_step])
                case DecodingStep.STEP_END:
                    current_step_queue.popleft()
                    if not current_step_queue:
                        break
                case DecodingStep.AI_NAME_VALUE:
                    current_trie_node = current_trie_node.children[
                        next_token_id
                    ]
                    if current_trie_node.is_end:
                        chosen_function = current_trie_node.function_name

                        func_spec = next(
                            f
                            for f in self.functions
                            if f.name == chosen_function
                        )
                        current_params_queue = list(
                            func_spec.parameters.items()
                        )
                        is_first_param = True

                        current_step = self.transitions[current_step]
                        current_step_queue = deque(steps_map[current_step])
                        current_trie_node = self.fn_name_trie.root
                case DecodingStep.AI_PARAMS_KEYS:
                    pass
                case DecodingStep.AI_PARAMS_VALS:
                    pass

        return {"raw": self.model.decode(generated_tokens)}
