from pathlib import Path

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    question: str
    reference_answer: str
    # Source files that contain the answer. Empty for deliberately unanswerable questions.
    relevant_sources: list[str] = Field(default_factory=list)


def load_dataset(path: Path) -> list[EvalExample]:
    """Read a JSONL file with one `EvalExample` per line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalExample.model_validate_json(line) for line in lines if line.strip()]
