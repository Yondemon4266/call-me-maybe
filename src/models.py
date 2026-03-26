"""Data models used by the constrained generation pipeline."""

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
)
import os
from typing import Literal, Any
from pathlib import Path


class PromptFormat(BaseModel):
    """Represent a single user prompt to process."""

    model_config = ConfigDict(extra="forbid")
    prompt: str


class TypeInfo(BaseModel):
    """Describe a supported JSON scalar type."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean", "integer"]


class FunctionFormat(BaseModel):
    """Define a callable function signature exposed to the decoder."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, TypeInfo]
    returns: TypeInfo


class FunctionCallOutput(BaseModel):
    """Represent one generated function call output."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]


class ProjectArgs(BaseModel):
    """Store validated CLI arguments for project execution."""

    model_config = ConfigDict(extra="forbid")
    functions_definition: Path = Field(...)
    input: Path = Field(...)
    output: Path = Field(...)

    @field_validator("input", "functions_definition")
    @classmethod
    def check_input_files(cls, value: Path) -> Path:
        """Validate that an input path exists, is a file, and is readable.

        Args:
            value: Path provided for an input JSON file.

        Returns:
            The same path when it points to an existing and readable file.

        Raises:
            ValueError: If the path does not exist, is not a file, or lacks
                read permissions.
        """
        if not value.is_file():
            raise ValueError(
                f"File '{value}'" "cannot be found or is not a file"
            )
        if not os.access(value, os.R_OK):
            raise ValueError(
                "Permission denied:" f" Cannot read the file '{value}'"
            )
        return value

    @field_validator("output")
    @classmethod
    def check_output_file(cls, value: Path) -> Path:
        """Validate write permissions for the output file or its parent
        directory.

        Args:
            value: Path provided for the output JSON file.

        Returns:
            The same path when write permissions are confirmed.

        Raises:
            ValueError: If the output path exists but is not a file, or if
                write permissions are denied for the file or its
                parent directory.
        """
        if value.exists():
            if not value.is_file():
                raise ValueError(
                    f"Output path '{value}' already exists and is not a file"
                )
            if not os.access(value, os.W_OK):
                raise ValueError(
                    "Permission denied: "
                    f"Cannot write to existing file '{value}'"
                )
        return value

    @model_validator(mode="after")
    def check_paths_are_unique(self) -> "ProjectArgs":
        """Ensure all configured paths are distinct after resolution.

        Returns:
            The current validated argument object.

        Raises:
            ValueError: If two or more resolved paths are identical.
        """
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
