"""Shared generation and analysis configuration for a repeated-answer study."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class BatchPrices(BaseModel):
    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write: float | None = Field(default=None, ge=0)


class StudyModel(BaseModel):
    id: str
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    route: Literal["anthropic", "openai", "openrouter"]
    host: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)?$")
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_mode: str = "default"
    max_tokens: int = Field(gt=0)
    batch_prices: BatchPrices | None = None


class StudyConfig(BaseModel):
    sample_prefixes: list[str] = Field(min_length=6, max_length=6)
    models: list[StudyModel] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_keys(self):
        for values in (
            self.sample_prefixes,
            [model.id for model in self.models],
            [model.alias for model in self.models],
        ):
            if len(set(values)) != len(values):
                raise ValueError("Study prefixes, model IDs, and aliases must each be unique")
        for index, prefix in enumerate(self.sample_prefixes):
            path = Path(prefix)
            if (path.is_absolute() or ".." in path.parts or path.as_posix() != prefix
                    or path.name != f"s{index:02d}"):
                raise ValueError(f"Invalid sample prefix {prefix!r}")
        return self

    @property
    def aliases(self) -> dict[str, str]:
        return {model.id: model.alias for model in self.models}


def load_study(path: Path) -> StudyConfig:
    return StudyConfig.model_validate(yaml.safe_load(path.read_text()))
