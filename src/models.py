from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
)

from typing import Literal, Any

from pathlib import Path


class PromptFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class TypeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean", "integer"]


class FunctionFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, TypeInfo]
    returns: TypeInfo


class FunctionCallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]


class ProjectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
