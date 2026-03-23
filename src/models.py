from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
)
from typing import Literal

from pathlib import Path


class PromptFormat(BaseModel):
    prompt: str


class TypeInfo(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class FunctionFormat(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeInfo]
    returns: TypeInfo


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
