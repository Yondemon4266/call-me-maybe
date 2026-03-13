import argparse
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ValidationError,
)
from typing import TypeVar, Type
import json
import sys


class TestPrompt(BaseModel):
    prompt: str


class TypeInfo(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeInfo]
    returns: TypeInfo


T = TypeVar("T", bound=BaseModel)


def load_and_validate_json(file_path: Path, model_class: Type[T]) -> list[T]:
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, list):
            sys.stderr.write(
                f"Error: The file {file_path} must contain a JSON array (list)."
            )
            sys.exit(1)

        for item in data:
            print(item)
        validated_data: list[T] = [
            model_class.model_validate(item) for item in data
        ]
        return validated_data

    except json.JSONDecodeError as e:
        sys.stderr.write(
            f"Error: {e.doc} does not contain valid JSON.\nDetails: "
            f"at line: {e.lineno}, col: {e.colno}",
        )
        sys.exit(1)
    except ValidationError as e:
        print(
            f"Error: The structure of {file_path} is invalid.\nDetails: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while reading {file_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


class ProjectArgs(BaseModel):
    functions_definition: Path = Field(...)
    input: Path = Field(...)
    output: Path = Field(...)

    @field_validator("input", "functions_definition")
    @classmethod
    def check_if_file_exists(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError("File can't be found or path is not a file")
        return value

    @model_validator(mode="after")
    def check_paths_are_unique(self) -> "ProjectArgs":
        paths: list[Path] = [
            self.functions_definition.resolve(),
            self.input.resolve(),
            self.output.resolve(),
        ]
        if len(set(paths)) != len(paths):
            raise ValueError(
                (
                    "There are some file paths that are identic,"
                    " they must all be different"
                )
            )
        return self


def parse_args() -> ProjectArgs:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Function calling tool with the AI model Qwen3-0.6B"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the functions definition file",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the file containing the test prompts",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to the generated JSON output file",
    )
    raw_args = parser.parse_args()
    try:
        args = ProjectArgs(**vars(raw_args))
        return args

    # check if input files exist
    except ValidationError as e:
        print(e)
        exit(1)
        # sys.stderr.write("error pydantic")


def main() -> None:
    """Point d'entrée principal du programme."""
    args = parse_args()

    print("Reading the files")
    # TODO: Ajouter ici la logique pour :
    # 1. Lire et valider les fichiers JSON d'entrée
    # 2. Initialiser le modèle Small_LLM_Model du llm_sdk
    # 3. Lancer la boucle de génération / constrained decoding
    # 4. Sauvegarder les résultats


if __name__ == "__main__":
    main()
