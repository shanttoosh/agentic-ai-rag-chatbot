FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Third-party dependencies (this layer is rebuilt only when requirements.txt changes).
COPY requirements.txt ./
RUN grep -v '^-e ' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# The project itself.
COPY pyproject.toml README.md ./
COPY app ./app
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY ui ./ui
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p data/raw data/processed evaluation/results \
    && chown -R appuser:appuser data evaluation/results

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
