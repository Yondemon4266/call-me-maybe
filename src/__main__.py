from llm_sdk import Small_LLM_Model
import src.parser as parser
from src.models import FunctionFormat, PromptFormat
from src.generation.decoder import JsonConstrainedDecoder
import json


def main() -> None:
    args = parser.parse_args()

    print("Reading the files...")

    prompts = parser.load_and_validate_json(args.input, PromptFormat)
    functions = parser.load_and_validate_json(
        args.functions_definition, FunctionFormat
    )
    ai_model = Small_LLM_Model()

    print("AI Model loaded with success!")
    decoder = JsonConstrainedDecoder(ai_model, prompts, functions)
    decoder.generate_all_prompts_in_json()


if __name__ == "__main__":
    main()
