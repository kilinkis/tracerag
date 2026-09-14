"""Domain models for source-backed NFL rule explanations.

Pydantic model and URL validation sources:
https://docs.pydantic.dev/latest/concepts/models/
https://docs.pydantic.dev/latest/api/networks/#pydantic.networks.HttpUrl
"""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SourceReference(BaseModel):
    """Authoritative material from which an explanation was derived."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    url: HttpUrl


class RuleDocument(BaseModel):
    """One versioned, plain-language rule explanation."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    season: int = Field(ge=1920)
    source: SourceReference
    rule_references: tuple[str, ...] = Field(min_length=1)
    content: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)


class RuleChunk(BaseModel):
    """A stable, independently retrievable section of a rule explanation."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    document_id: str
    document_title: str
    season: int
    heading_path: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    source: SourceReference
    rule_references: tuple[str, ...]
    relative_path: str
