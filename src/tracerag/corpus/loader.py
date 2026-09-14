"""Load and structurally chunk the version-controlled Markdown corpus."""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tracerag.corpus.models import RuleChunk, RuleDocument, SourceReference

_METADATA_DELIMITER = "+++"
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def load_document(path: Path, *, corpus_root: Path | None = None) -> RuleDocument:
    """Parse one Markdown file with a TOML metadata header."""

    raw = path.read_text(encoding="utf-8")
    metadata, content = _split_metadata(raw, path)
    root = corpus_root or path.parent

    return RuleDocument(
        document_id=_required(metadata, "document_id", path),
        title=_required(metadata, "title", path),
        season=_required(metadata, "season", path),
        source=SourceReference(
            title=_required(metadata, "source_title", path),
            url=_required(metadata, "source_url", path),
        ),
        rule_references=tuple(_required(metadata, "rule_references", path)),
        content=content.strip(),
        relative_path=path.relative_to(root).as_posix(),
    )


def load_corpus(corpus_root: Path) -> list[RuleDocument]:
    """Load every corpus document in stable path order."""

    documents = [
        load_document(path, corpus_root=corpus_root)
        for path in sorted(corpus_root.glob("*.md"))
        if path.name != "README.md"
    ]
    _validate_corpus(documents, corpus_root)
    return documents


def chunk_document(document: RuleDocument, *, max_chars: int = 1_200) -> list[RuleChunk]:
    """Create deterministic chunks at Markdown heading and paragraph boundaries."""

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    chunks: list[RuleChunk] = []
    for heading_path, section_text in _iter_sections(document.content):
        for part_number, text in enumerate(_split_section(section_text, max_chars), start=1):
            identity = "\n".join(
                (
                    str(document.season),
                    document.document_id,
                    *heading_path,
                    str(part_number),
                    text,
                )
            )
            chunks.append(
                RuleChunk(
                    chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                    document_id=document.document_id,
                    document_title=document.title,
                    season=document.season,
                    heading_path=heading_path,
                    text=text,
                    source=document.source,
                    rule_references=document.rule_references,
                    relative_path=document.relative_path,
                )
            )

    if not chunks:
        raise ValueError(f"{document.relative_path}: document has no headed sections")
    return chunks


def chunk_corpus(
    documents: Iterable[RuleDocument], *, max_chars: int = 1_200
) -> list[RuleChunk]:
    """Chunk a sequence of documents while preserving document order."""

    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, max_chars=max_chars)
    ]


def _split_metadata(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    if not lines or lines[0] != _METADATA_DELIMITER:
        raise ValueError(f"{path}: expected TOML metadata header")

    try:
        closing_index = lines[1:].index(_METADATA_DELIMITER) + 1
    except ValueError as error:
        raise ValueError(f"{path}: metadata header is not closed") from error

    metadata_text = "\n".join(lines[1:closing_index])
    try:
        metadata = tomllib.loads(metadata_text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: invalid TOML metadata: {error}") from error

    return metadata, "\n".join(lines[closing_index + 1 :])


def _required(metadata: dict[str, Any], key: str, path: Path) -> Any:
    try:
        return metadata[key]
    except KeyError as error:
        raise ValueError(f"{path}: missing metadata field {key!r}") from error


def _validate_corpus(documents: list[RuleDocument], corpus_root: Path) -> None:
    if not documents:
        raise ValueError(f"{corpus_root}: no corpus documents found")

    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{corpus_root}: duplicate document_id")

    seasons = {document.season for document in documents}
    if len(seasons) != 1:
        raise ValueError(f"{corpus_root}: expected exactly one corpus season")


def _iter_sections(content: str) -> Iterable[tuple[tuple[str, ...], str]]:
    heading_stack: list[str] = []
    body: list[str] = []

    for line in content.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match:
            if heading_stack and any(part.strip() for part in body):
                yield tuple(heading_stack), "\n".join(body).strip()
            level = len(match.group(1))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(match.group(2))
            body = []
        elif heading_stack:
            body.append(line)
        elif line.strip():
            raise ValueError("document content must begin with a Markdown heading")

    if heading_stack and any(part.strip() for part in body):
        yield tuple(heading_stack), "\n".join(body).strip()


def _split_section(text: str, max_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    parts: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            raise ValueError("a corpus paragraph exceeds max_chars and must be edited")
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = paragraph

    if current:
        parts.append(current)
    return parts
