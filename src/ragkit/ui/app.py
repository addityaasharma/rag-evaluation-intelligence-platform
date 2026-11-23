from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from ragkit.config import Settings, get_settings
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
from ragkit.models import Answer, ChatMessage
from ragkit.pipeline import RagPipeline, build_pipeline

DATASET = Path("eval/datasets/golden.jsonl")
HEADLINE_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Relevancy",
    "precision": "Precision",
    "recall": "Recall",
}


SAMPLE_DOCS = Path("data/sample_docs")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


@st.cache_resource
def _shared_pipeline() -> RagPipeline:
    return build_pipeline()


@st.cache_resource(max_entries=20, ttl=3600)
def _demo_pipeline(session_id: str) -> RagPipeline:
    settings = get_settings()
    demo = Settings(**{**settings.model_dump(), "collection": f"demo-{session_id}"})
    instance = build_pipeline(demo)
    instance.ingest(SAMPLE_DOCS, reset=True)
    return instance


def pipeline() -> RagPipeline:
    """The shared index, or in public demo mode a private one per browser session."""
    if not get_settings().public_demo:
        return _shared_pipeline()
    return _demo_pipeline(st.session_state.setdefault("session_id", uuid4().hex[:10]))


@st.cache_resource
def history() -> HistoryStore:
    return HistoryStore(get_settings().history_path)


def documents_page() -> None:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True
    )
    if uploads and st.button("Index uploads", type="primary"):
        for upload in uploads:
            if get_settings().public_demo and upload.size > MAX_UPLOAD_BYTES:
                st.error(f"{upload.name}: demo uploads are limited to 5 MB")
                continue
            try:
                with st.spinner(f"Indexing {upload.name}"):
                    stats = pipeline().ingest_upload(upload.name, upload.getvalue())
            except Exception as exc:
                st.error(f"{upload.name}: {exc}")
                continue
            if stats.documents:
                st.success(f"{upload.name}: {stats.chunks} chunks")
            else:
                st.warning(f"{upload.name}: no extractable text (scanned PDF?)")

    st.subheader("Indexed")
    docs = pipeline().documents()
    if not docs:
        st.info("No documents indexed yet.")
    for doc in docs:
        name_col, count_col, action_col = st.columns([6, 2, 2])
        name_col.write(doc.source)
        count_col.write(f"{doc.chunks} chunks")
        if action_col.button("Delete", key=f"delete-{doc.source}"):
            pipeline().delete_document(doc.source)
            st.rerun()


def render_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    if answer.verification:
        if answer.verification.has_hallucination:
            st.warning(
                "Possible hallucination, unsupported claims:\n"
                + "\n".join(f"- {c.text}" for c in answer.verification.unsupported)
            )
        else:
            st.success("All claims are supported by the retrieved context.")
    cited = set(answer.cited_indices)
    with st.expander(f"Sources ({len(answer.contexts)})"):
        for i, ctx in enumerate(answer.contexts, start=1):
            page = f", page {ctx.chunk.page}" if ctx.chunk.page else ""
            marker = " (cited)" if i in cited else ""
            st.markdown(f"**[{i}] {ctx.chunk.source}{page}**{marker} · score {ctx.score:.3f}")
            st.caption(ctx.chunk.text)


def chat_page() -> None:
    st.header("Chat")
    verify = st.sidebar.toggle("Check for hallucinations", value=False)
    messages: list[tuple[ChatMessage, Answer | None]] = st.session_state.setdefault("chat", [])

    for message, answer in messages:
        with st.chat_message(message.role):
            render_answer(answer) if answer else st.markdown(message.content)

    settings = get_settings()
    asked = st.session_state.get("asked", 0)
    if settings.public_demo and asked >= settings.demo_question_limit:
        st.info("Demo question limit reached for this session. Run it locally for unlimited use.")
        return
    question = st.chat_input("Ask about your documents")
    if not question:
        return
    st.session_state["asked"] = asked + 1
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            answer = pipeline().ask(question, history=[m for m, _ in messages], verify=verify)
        render_answer(answer)
    messages.append((ChatMessage(role="user", content=question), None))
    messages.append((ChatMessage(role="assistant", content=answer.text), answer))


def _run_table(runs: list) -> pd.DataFrame:  # type: ignore[type-arg]
    return pd.DataFrame(
        [
            {
                "run": r.id,
                "when": r.created_at,
                "label": r.label,
                "chunk": r.config.get("chunk_size"),
                "embedding": r.config.get("embedding_model"),
                "mode": r.config.get("retrieval_mode"),
                **r.summary,
            }
            for r in runs
        ]
    )


def evaluation_page() -> None:
    st.header("Evaluation")
    settings = get_settings()
    dashboard, compare = st.tabs(["Dashboard", "Compare configurations"])
    with dashboard:
        _dashboard(settings)
    with compare:
        _compare(settings)


def _eval_used_up(settings: Settings) -> bool:
    # Each evaluation costs dozens of LLM calls, so the public demo allows one per session.
    return settings.public_demo and st.session_state.get("evals", 0) >= 1


def _dashboard(settings: Settings) -> None:
    label = st.text_input("Run label", value="ui")
    if st.button("Run evaluation on golden dataset", disabled=_eval_used_up(settings)):
        st.session_state["evals"] = st.session_state.get("evals", 0) + 1
        with st.spinner("Evaluating (makes LLM calls)"):
            judge = LLMJudge(LiteLLMClient(settings.judge_model, max_tokens=256))
            report = run_evaluation(pipeline(), load_dataset(DATASET), judge, workers=2)
            history().record(label, config_snapshot(settings, DATASET.name), report)

    runs = history().runs()
    if not runs:
        st.info("No evaluation runs yet.")
        return

    latest = runs[0]
    st.caption(f"Run #{latest.id} · {latest.label} · {latest.created_at}")
    for column, (key, title) in zip(st.columns(4), HEADLINE_METRICS.items(), strict=True):
        previous = runs[1].summary.get(key) if len(runs) > 1 else None
        value = latest.summary.get(key)
        delta = None if value is None or previous is None else f"{value - previous:+.3f}"
        column.metric(title, "n/a" if value is None else f"{value:.3f}", delta)

    report = history().report(latest.id)
    rows = pd.DataFrame(
        [
            {
                "question": e.question,
                "faithfulness": e.scores.get("faithfulness"),
                "possible hallucination": e.possible_hallucination,
                "sources": ", ".join(e.retrieved_sources),
                "answer": e.answer,
            }
            for e in report.examples
        ]
    )
    flagged = int(rows["possible hallucination"].sum())
    st.subheader(f"Per-question results ({flagged} possible hallucinations)")
    st.dataframe(rows, use_container_width=True)

    st.subheader("History")
    table = _run_table(runs)
    metrics = [m for m in HEADLINE_METRICS if m in table.columns]
    if metrics:
        st.line_chart(table.sort_values("run").set_index("run")[metrics])
    st.dataframe(table, use_container_width=True)


def _compare(settings: Settings) -> None:
    st.write(
        "Index the same documents with each combination and score retrieval. "
        "No LLM calls unless answer scoring is enabled."
    )
    docs = st.text_input("Documents folder", value="data/sample_docs")
    sizes = st.text_input("Chunk sizes", value="400,800")
    models = st.text_input("Embedding models", value=settings.embedding_model)
    modes = st.multiselect("Retrieval modes", ["dense", "bm25", "hybrid"], default=["hybrid"])
    with_judges = st.checkbox("Also score answer quality (LLM calls)")
    if not st.button("Run comparison"):
        return

    grid = SweepGrid(
        chunk_sizes=[int(s) for s in sizes.split(",") if s.strip()],
        embedding_models=[m.strip() for m in models.split(",") if m.strip()],
        retrieval_modes=modes,  # type: ignore[arg-type]
    )
    judge = LLMJudge(LiteLLMClient(settings.judge_model, max_tokens=256)) if with_judges else None
    with st.spinner("Running sweep"):
        results = run_sweep(
            settings, Path(docs), load_dataset(DATASET), grid, judge=judge, history=history()
        )
    st.dataframe(
        pd.DataFrame([{**r.config, **r.summary} for r in results]), use_container_width=True
    )


st.set_page_config(page_title="RAG Evaluation & Intelligence Platform", layout="wide")
st.navigation(
    [
        st.Page(chat_page, title="Chat", url_path="chat", default=True),
        st.Page(documents_page, title="Documents", url_path="documents"),
        st.Page(evaluation_page, title="Evaluation", url_path="evaluation"),
    ]
).run()
