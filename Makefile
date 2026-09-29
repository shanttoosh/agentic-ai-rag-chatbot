# Make targets (Linux/macOS/WSL/Git Bash). README.md lists the equivalent direct commands.
PYTHON ?= python

.PHONY: install ingest api ui test test-live lint format typecheck check evaluate docker-up

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

ingest:
	$(PYTHON) scripts/ingest.py

api:
	uvicorn app.main:app --reload

ui:
	streamlit run ui/streamlit_app.py

test:
	$(PYTHON) -m pytest

test-live:
	RUN_LIVE_TESTS=1 $(PYTHON) -m pytest tests/integration/test_pinecone.py

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

check: lint typecheck test

evaluate:
	$(PYTHON) scripts/evaluate.py

docker-up:
	docker compose up --build
