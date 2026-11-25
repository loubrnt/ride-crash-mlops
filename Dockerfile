# API image. The model is exported from the MLflow Registry before the build:
#   uv run python -m crashdetect.export_model   # Registry @production -> ./model
#   docker build -t loubrunet/ride-crash-api .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lock file changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY README.md params.yaml ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

COPY model ./model

RUN useradd --create-home appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=/app/model
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "crashdetect.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
