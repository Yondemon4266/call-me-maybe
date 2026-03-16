from llm_sdk import Small_LLM_Model
from src.models import FunctionFormat, PromptFormat
from pydantic import BaseModel
from enum import Enum


class DecodingStep(Enum):
    STEP_START_TO_PROMPT = 1  # Forces: {\n  "prompt": "
    STEP_PROMPT_VALUE = 2  # Forces: prompt exact value
    STEP_TO_NAME = 3  # Forces: ",\n  "name": "

    AI_NAME_VALUE = 4  # AI Choice: fn_add_numbers or fn_greet...

    STEP_TO_PARAMS = 5  # Forces: ",\n  "parameters": {

    AI_PARAMS_KEYS = 6  # AI Choice: params via the function chosen
    AI_PARAMS_VALS = 7  # AI Choice: filter per type (number, string, boolean)

    STEP_END = 8  # Forces: }\n}


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
        self.context = "You are a helpful assistant. Choose the correct "
        "function and get the needed arguments "
        "based on the user prompt for it to work.\n"
        self.vocabulary_file = model.get_path_to_vocab_file()
        self.step_start_ids = self.model.encode('{\n  "prompt": "').tolist()[0]
        self.step_to_name_ids = self.model.encode('",\n  "name": "').tolist()[
            0
        ]
        self.transitions = {
            DecodingStep.STEP_START_TO_PROMPT: DecodingStep.STEP_PROMPT_VALUE,
            DecodingStep.STEP_PROMPT_VALUE: DecodingStep.STEP_TO_NAME,
            DecodingStep.STEP_TO_NAME: DecodingStep.AI_NAME_VALUE,
            # La suite viendra s'ajouter ici naturellement
        }

    def generate_json_in_output_format(self, prompt: "PromptFormat") -> dict:
        print(f"\n--- [PURISTE] Traitement de : {prompt.prompt} ---")

        input_ids = self.model.encode(self.context).tolist()[0]

        prompt_steps = {
            DecodingStep.STEP_START_TO_PROMPT: self.step_start_ids,
            DecodingStep.STEP_PROMPT_VALUE: self.model.encode(
                prompt.prompt
            ).tolist()[0],
            DecodingStep.STEP_TO_NAME: self.step_to_name_ids,
        }

        current_step = DecodingStep.STEP_START_TO_PROMPT
        current_step_queue = list(prompt_steps[current_step])

        max_tokens = 200
        generated_tokens = []

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_token_ids = []

            if current_step in prompt_steps:
                allowed_token_ids = [current_step_queue[0]]

            elif current_step == DecodingStep.AI_NAME_VALUE:
                # Mode Aiguillage : On devra demander à notre Trie les tokens possibles
                print("\n[Arrivé à l'aiguillage du nom de fonction !]")
                break

            if allowed_token_ids:
                for i in range(len(logits)):
                    if i not in allowed_token_ids:
                        logits[i] = -float("inf")

            next_token_id = logits.index(max(logits))
            input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)
            print(self.model.decode([next_token_id]), end="", flush=True)

            if current_step in prompt_steps:
                current_step_queue.pop(0)

                # Le step est fini ? On passe à l'état suivant !
                if len(current_step_queue) == 0:
                    current_step = self.transitions[current_step]

                    # Si le NOUVEL état est aussi un step, on charge sa file d'attente
                    if current_step in prompt_steps:
                        current_step_queue = list(prompt_steps[current_step])

        return {"raw": self.model.decode(generated_tokens)}
