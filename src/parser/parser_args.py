"""CLI argument parsing and validation entrypoints."""

import argparse
from src.models import ProjectArgs
from pathlib import Path
from pydantic import ValidationError
import sys


def parse_args() -> ProjectArgs:
    """Parse CLI arguments and validate them with `ProjectArgs`.

    Returns:
        A validated `ProjectArgs` instance.

    Raises:
        SystemExit: If argument validation fails.
    """
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

    except ValidationError as e:
        sys.stderr.write(
            f"{e.errors()[0]['msg']} \nInput was: {e.errors()[0]['input']}"
        )
        exit(1)
