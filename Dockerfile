FROM python:3.12-slim

RUN useradd -m -u 1000 user
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

USER user
ENV HOME=/home/user \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    RAG_PUBLIC_DEMO=0 \
    RAG_CHROMA_PATH=/home/user/data/chroma \
    RAG_HISTORY_PATH=/home/user/data/history.db
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --all-extras --no-install-project

COPY --chown=user . .
RUN uv sync --frozen --all-extras \
    && uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/ragkit/ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
