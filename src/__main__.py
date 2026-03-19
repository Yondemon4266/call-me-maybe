from llm_sdk import Small_LLM_Model
import src.parser as parser
from src.models import FunctionFormat, PromptFormat
from src.generation.json_types.type_registry import JSONTypeRegistry
from src.generation.decoder import JsonConstrainedDecoder
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
    # print(ai_model.get_path_to_tokenizer_file())
    decoder = JsonConstrainedDecoder(ai_model, prompts, functions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("[\n")

    is_first = True  # Pour savoir si on doit mettre une virgule avant l'objet
    for prompt in prompts:
        json_str = decoder.generate_json_in_output_format(prompt)
        try:
            parsed_json = json.loads(json_str)
            with open(args.output, "a", encoding="utf-8") as f:
                if not is_first:
                    f.write(
                        ",\n"
                    )  # Ajoute une virgule avant, sauf pour le tout premier

                # Écrit l'objet JSON proprement avec une indentation
                json.dump(parsed_json, f, indent=2)
            is_first = False
        except json.JSONDecodeError as e:
            # Le sujet exige que le programme ne crash pas et gère les erreurs gracieusement
            print(
                f"\n[Erreur] L'IA a généré un JSON invalide pour : '{prompt.prompt}'"
            )
            print(f"Détails : {e}")
            continue

    with open(args.output, "a", encoding="utf-8") as f:
        f.write("\n]\n")
    print(f"\nTerminé ! Les résultats ont été sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
