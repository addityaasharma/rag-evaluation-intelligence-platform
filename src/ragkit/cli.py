import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ragkit.config import get_settings
from ragkit.evaluation import (
    HistoryStore,
    LLMJudge,
    SweepGrid,
    config_snapshot,
    load_dataset,
    run_evaluation,
    run_sweep,
)
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
def ask(
    question: str,
    top_k: Annotated[int | None, typer.Option()] = None,
    verify: Annotated[bool, typer.Option(help="Check the answer for unsupported claims.")] = False,
) -> None:
    """Answer a question from the indexed documents."""
    answer = build_pipeline().ask(question, top_k, verify=verify)
    console.print(answer.text)
    for i, ctx in enumerate(answer.contexts, start=1):
        page = f", p.{ctx.chunk.page}" if ctx.chunk.page else ""
        console.print(f"[dim][{i}] {ctx.chunk.source}{page} (score {ctx.score:.3f})[/dim]")
    if answer.verification:
        for claim in answer.verification.unsupported:
            console.print(f"[yellow]Unsupported claim:[/yellow] {claim.text}")


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
    label: Annotated[str, typer.Option(help="Name for this run in the history.")] = "eval",
    record: Annotated[bool, typer.Option(help="Save the run to the history database.")] = True,
) -> None:
    """Score retrieval and answer quality; exit non-zero if a --min-score gate fails."""
    thresholds = _parse_thresholds(min_score or [])
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    judge = LLMJudge(LiteLLMClient(settings.judge_model, max_tokens=256))
    report = run_evaluation(build_pipeline(settings), load_dataset(dataset), judge, workers=workers)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    if record:
        run_id = HistoryStore(settings.history_path).record(
            label, config_snapshot(settings, dataset.name), report
        )
        console.print(f"Recorded as run #{run_id}.")
    _print_summary(report.summary, thresholds)

    if failing := report.failing(thresholds):
        console.print(f"[red]Gate failed:[/red] {failing}")
        raise typer.Exit(code=1)


@app.command()
def history(limit: Annotated[int, typer.Option(min=1)] = 20) -> None:
    """List recent evaluation runs."""
    runs = HistoryStore(get_settings().history_path).runs(limit)
    metrics = sorted({m for run in runs for m in run.summary})
    table = Table(title="Evaluation history")
    for column in ("Run", "When", "Label", "Chunk", "Embedding", "Mode", *metrics):
        table.add_column(column)
    for run in runs:
        c = run.config
        table.add_row(
            str(run.id),
            run.created_at,
            run.label,
            str(c.get("chunk_size")),
            str(c.get("embedding_model")),
            str(c.get("retrieval_mode")),
            *(f"{run.summary[m]:.3f}" if m in run.summary else "-" for m in metrics),
        )
    console.print(table)


@app.command()
def compare(
    docs: Annotated[Path, typer.Argument(help="Documents to index for every configuration.")],
    chunk_size: Annotated[str, typer.Option(help="Comma-separated sizes.")] = "400,800",
    embedding_model: Annotated[str, typer.Option(help="Comma-separated models.")] = "",
    retrieval_mode: Annotated[str, typer.Option(help="Comma-separated modes.")] = "hybrid",
    dataset: Annotated[Path, typer.Option()] = Path("eval/datasets/golden.jsonl"),
    with_judges: Annotated[
        bool, typer.Option(help="Also score answer quality (makes LLM calls).")
    ] = False,
) -> None:
    """Compare chunk sizes, embedding models and retrieval modes on one dataset."""
    settings = get_settings()
    grid = SweepGrid(
        chunk_sizes=[int(v) for v in chunk_size.split(",")],
        embedding_models=[v for v in embedding_model.split(",") if v] or [settings.embedding_model],
        retrieval_modes=[v.strip() for v in retrieval_mode.split(",")],  # type: ignore[misc]
    )
    judge = LLMJudge(LiteLLMClient(settings.judge_model, max_tokens=256)) if with_judges else None
    results = run_sweep(
        settings,
        docs,
        load_dataset(dataset),
        grid,
        judge=judge,
        history=HistoryStore(settings.history_path),
    )

    metrics = sorted({m for r in results for m in r.summary})
    table = Table(title="Configuration comparison")
    for column in ("Chunk", "Embedding", "Mode", *metrics):
        table.add_column(column)
    for r in results:
        c = r.config
        table.add_row(
            str(c["chunk_size"]),
            c["embedding_model"],
            c["retrieval_mode"],
            *(f"{r.summary[m]:.3f}" if m in r.summary else "-" for m in metrics),
        )
    console.print(table)


@app.command()
def ui(port: int = 8501) -> None:
    """Launch the Streamlit app (requires the `ui` extra)."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).parent / "ui" / "app.py"),
            "--server.port",
            str(port),
        ],
        check=False,
    )


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
