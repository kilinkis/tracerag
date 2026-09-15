"""Load and validate version-controlled evaluation cases."""

from pathlib import Path

from tracerag.evaluation.models import EvaluationDataset


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """Return a validated evaluation dataset from one JSON file."""

    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
