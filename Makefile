.DEFAULT_GOAL := help
API_IMAGE := loubrunet/ride-crash-api
APP_IMAGE := loubrunet/ride-crash-app
export MLFLOW_DISABLE_AGENT_HINT := 1

help: ## Command list
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Install the Python environment (uv sync), once per clone
	uv sync --all-groups

# --- Model lifecycle ---------------------------------------------------------
pipeline: ## Full DVC pipeline: features -> split -> train -> evaluate -> export
	uv run dvc repro

train: ## Train and register a new version in MLflow
	uv run python -m crashdetect.train

evaluate: ## Gate on the test set, promote to @production if it passes
	uv run python -m crashdetect.evaluate

export: ## Download the @production model to ./model
	uv run python -m crashdetect.export_model

mlflow: ## MLflow UI on http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# --- Code quality ------------------------------------------------------------
lint: ## Ruff (lint + format check)
	uv run ruff check src tests app && uv run ruff format --check src tests app

test: ## Unit, integration and API tests
	uv run pytest -q

e2e: ## End-to-end tests against a running API (default http://localhost:8000)
	API_URL=$${API_URL:-http://localhost:8000} uv run pytest -q -m e2e

# --- Serving -----------------------------------------------------------------
api: ## Local API on http://localhost:8000 (Swagger: /docs)
	MODEL_URI=model uv run uvicorn crashdetect.api.main:app --reload --port 8000

app: ## Local web app on http://localhost:8501
	API_URL=http://localhost:8000 uv run streamlit run app/streamlit_app.py

images: ## Build both Docker images (needs ./model: make export)
	docker build -t $(API_IMAGE):latest .
	docker build -f app/Dockerfile -t $(APP_IMAGE):latest .

up: ## Run API + web app with Docker Compose
	docker compose up -d

down: ## Stop Docker Compose
	docker compose down

.PHONY: help setup pipeline train evaluate export mlflow lint test e2e api app images up down
