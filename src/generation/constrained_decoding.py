from llm_sdk import Small_LLM_Model
from src.models import FunctionFormat, PromptFormat
from pydantic import BaseModel


class JsonStage(BaseModel):
    OPEN_CURLY_BRACKET
    



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
        self.static_elements = ["{", "}", '"', "prompt", "name", "parameters"]
        self.static_token_ids: dict[str, int] = self.init_static_tokens()

    def generate_json_in_output_format(self, prompt: PromptFormat):
        pass

    def init_static_tokens(self) -> dict:
        static_token_ids: dict[str, int] = {}
        for element in self.static_elements:
            static_token_ids[element] = self.model.encode(element).tolist()[0]
        return static_token_ids
