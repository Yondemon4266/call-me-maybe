from llm_sdk import Small_LLM_Model
import src.parser as parser
from src.models import FunctionDefinition, TestPrompt


def main() -> None:
    """Point d'entrée principal du programme."""
    args = parser.parse_args()

    print("Reading the files")

    functions = parser.load_and_validate_json(
        args.functions_definition, FunctionDefinition
    )
    prompts = parser.load_and_validate_json(args.input, TestPrompt)

    # TODO: Ajouter ici la logique pour:
    # 1. Lire et valider les fichiers JSON d'entrée - OK
    ai_model = Small_LLM_Model()
    ai_model.encode
    # ai_model.
    # 2. Initialiser le modèle Small_LLM_Model du llm_sdk - EN COURS
    # 3. Lancer la boucle de génération / constrained decoding
    # 4. Sauvegarder les résultats


if __name__ == "__main__":
    main()
