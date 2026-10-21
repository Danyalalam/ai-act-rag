# Container image for the AI Act RAG web app (FastAPI). The models are NOT in here:
# the app talks to Ollama over HTTP (OLLAMA_BASE_URL), and to Qdrant Cloud.
FROM python:3.12-slim

# uv, copied from its official image (same version as used locally).
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1. Install dependencies only. This layer is cached and only rebuilt when
#    pyproject.toml or uv.lock change, not on every code change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Copy the code and install the project itself.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Run as an unprivileged user, not root.
RUN useradd --create-home app && chown -R app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 0.0.0.0 = accept connections from outside the container (localhost would not).
CMD ["uvicorn", "ai_act_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
