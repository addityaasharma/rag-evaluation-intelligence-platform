import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ragkit.config import get_settings
from ragkit.evaluation import LLMJudge, load_dataset, run_evaluation
from ragkit.llm import LiteLLMClient
from ragkit.pipeline import build_pipeline

app = typer.Typer(help="Retrieval-augmented generation toolkit.", no_args_is_help=True)
console = Console()


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory of .md/.txt documents.")],
    reset: Annotated[bool, typer.Option(help="Clear the collection first.")] = False,
) -> None:
    """Chunk, embed and index documents."""
    stats = build_pipeline().ingest(path, reset=reset)
    console.print(f"Indexed {stats.chunks} chunks from {stats.documents} documents.")


@app.command()
def ask(question: str, top_k: Annotated[int | None, typer.Option()] = None) -> None:
    """Answer a question from the indexed documents."""
    answer = build_pipeline().ask(question, top_k)
    console.print(answer.text)
    for i, ctx in enumerate(answer.contexts, start=1):
        console.print(f"[dim][{i}] {ctx.chunk.source} (score {ctx.score:.3f})[/dim]")


@app.command("eval")
def evaluate(
    dataset: Annotated[Path, typer.Option(help="JSONL evaluation set.")] = Path(
        "eval/datasets/golden.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Where to write the JSON report.")] = Path(
        "eval/reports/latest.json"
    ),
    min_score: Annotated[
        list[str] | None,
        typer.Option(
            "--min-score", help="Fail if a metric mean is below it, e.g. faithfulness=0.8"
        ),
    ] = None,
    workers: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
    """Score retrieval and answer quality; exit non-zero if a --min-score gate fails."""
    thresholds = _parse_thresholds(min_score or [])
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    judge = LLMJudge(LiteLLMClient(settings.judge_model, max_tokens=256))
    report = run_evaluation(build_pipeline(settings), load_dataset(dataset), judge, workers=workers)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    _print_summary(report.summary, thresholds)

    if failing := report.failing(thresholds):
        console.print(f"[red]Gate failed:[/red] {failing}")
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("ragkit.api:create_app", host=host, port=port, factory=True)


def _parse_thresholds(pairs: list[str]) -> dict[str, float]:
    try:
        return {name: float(value) for name, value in (p.split("=", 1) for p in pairs)}
    except ValueError as exc:
        raise typer.BadParameter("Use the form metric=value, e.g. faithfulness=0.8") from exc


def _print_summary(summary: dict[str, float], thresholds: dict[str, float]) -> None:
    table = Table(title="Evaluation summary")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("Gate", justify="right")
    for name, value in summary.items():
        gate = thresholds.get(name)
        table.add_row(name, f"{value:.3f}", "" if gate is None else f">= {gate:.2f}")
    console.print(table)
