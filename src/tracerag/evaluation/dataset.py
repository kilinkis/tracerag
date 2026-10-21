"""Load and validate version-controlled evaluation cases."""

import hashlib
from pathlib import Path

from tracerag.evaluation.models import EvaluationDataset


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """Return a validated evaluation dataset from one JSON file."""

    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def fingerprint_dataset(dataset: EvaluationDataset) -> str:
    """Bind generated outputs and later reviews to the same questions and rubrics."""
    return hashlib.sha256(dataset.model_dump_json().encode()).hexdigest()
