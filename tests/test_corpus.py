"""Tests for deterministic NFL corpus ingestion."""

from pathlib import Path

import pytest

from tracerag.corpus.loader import chunk_corpus, chunk_document, load_corpus, load_document

CORPUS_ROOT = Path(__file__).parents[1] / "data" / "nfl"


def test_loads_versioned_corpus_in_stable_order() -> None:
    documents = load_corpus(CORPUS_ROOT)

    assert len(documents) == 6
    assert [document.document_id for document in documents] == [
        "clock-runoffs",
        "completing-a-catch",
        "downs-and-series",
        "passes-and-fumbles",
        "receiver-eligibility",
        "scoring",
    ]
    assert {document.season for document in documents} == {2026}
    assert all(document.rule_references for document in documents)


def test_chunk_ids_and_order_are_deterministic() -> None:
    documents = load_corpus(CORPUS_ROOT)

    first = chunk_corpus(documents)
    second = chunk_corpus(documents)

    assert first == second
    assert len(first) == 21
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert first[0].heading_path == (
        "Clock Management and Ten-Second Runoffs",
        "Illegally conserving time",
    )


def test_long_sections_split_only_between_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "example.md"
    path.write_text(
        """+++
document_id = "example"
title = "Example"
season = 2026
source_title = "Official source"
source_url = "https://example.com/rules"
rule_references = ["1-1-1"]
+++

# Example

## Section

First paragraph has enough text to stand on its own within the configured chunk size.

Second paragraph also has enough text to force a new chunk without splitting either paragraph.
""",
        encoding="utf-8",
    )

    document = load_document(path)
    chunks = chunk_document(document, max_chars=200)

    assert len(chunks) == 1
    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("First paragraph")
    assert chunks[1].text.startswith("Second paragraph")


def test_rejects_document_without_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid.md"
    path.write_text("# Missing metadata\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected TOML metadata header"):
        load_document(path)
