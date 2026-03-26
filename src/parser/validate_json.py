"""Helpers to load JSON files and validate items with Pydantic models."""

from typing import TypeVar, Type
import json
from pydantic import (
    BaseModel,
    ValidationError,
)
from pathlib import Path
import sys


T = TypeVar("T", bound=BaseModel)


def load_and_validate_json(file_path: Path, model_class: Type[T]) -> list[T]:
    """Load a JSON list file and validate each item with a model class.

    Args:
        file_path: Path to the JSON file to read.
        model_class: Pydantic model class used to validate each list item.

    Returns:
        A list of validated model instances.

    Raises:
        SystemExit: If file content is invalid, malformed, or unreadable.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, list):
            sys.stderr.write(
                f"Error: The file {file_path} "
                "must contain a JSON array (list)"
            )
            sys.exit(1)

        validated_data: list[T] = [
            model_class.model_validate(item) for item in data
        ]
        return validated_data

    except json.JSONDecodeError as e:
        sys.stderr.write(
            f"Error: {file_path} does not contain valid JSON.\nDetails: "
            f"{e.msg} at line: {e.lineno}, col: {e.colno}",
        )
        sys.exit(1)
    except ValidationError as e:
        sys.stderr.write(
            f"Error: The structure of {file_path} is invalid.\nDetails: {e}"
        )
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(
            f"An unexpected error occurred while reading {file_path}: {e}"
        )
        sys.exit(1)
