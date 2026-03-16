import argparse
from src.models import ProjectArgs
from pathlib import Path
from pydantic import ValidationError


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
