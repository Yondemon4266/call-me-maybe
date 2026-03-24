from llm_sdk import Small_LLM_Model
import src.parser as parser
from src.models import FunctionFormat, PromptFormat, ProjectArgs
from src.generation.decoder import JsonConstrainedDecoder
import os


def write_to_output(json_output: str, args: ProjectArgs) -> None:
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(json_output)
    except OSError as e:
        print(e)


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
    json_output: str = decoder.generate_all_prompts_in_json()
    write_to_output(json_output, args)


if __name__ == "__main__":
    main()
