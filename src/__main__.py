from llm_sdk import Small_LLM_Model
import src.parser as parser
from src.models import FunctionFormat, PromptFormat
from src.generation.constrained_decoding import JsonConstrainedDecoder
import json


def main() -> None:
    """Point d'entrée principal du programme."""
    args = parser.parse_args()

    print("Reading the files")

    prompts = parser.load_and_validate_json(args.input, PromptFormat)
    functions = parser.load_and_validate_json(
        args.functions_definition, FunctionFormat
    )
    print("Initializing the AI model Qwen/Qwen3-0.6B...")
    ai_model = Small_LLM_Model()
    print(f"AI Model loaded with success on the {ai_model._device}!")

    decoder = JsonConstrainedDecoder(ai_model, prompts, functions)
    # 3. Lancer la boucle de génération / constrained decoding - EN COURS
    # 4. Sauvegarder les résultats


if __name__ == "__main__":
    main()
