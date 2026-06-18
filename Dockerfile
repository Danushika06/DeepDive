# DeepDive — multi-stage build.
# Stage 1 installs deps (CPU-only torch to keep the image ~1.7 GB vs ~8 GB CUDA).
# Stage 2 is a slim runtime that pre-downloads the embedding model at build time so the
# first request pays no download cost and the container can run offline.

# ---------- builder ----------
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY requirements-cpu.txt requirements.txt ./
# CPU torch first (separate index), then everything else against PyPI.
RUN pip install --no-cache-dir -r requirements-cpu.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Bake the embedding model into the image (depends only on site-packages, so this layer
# is cached across application-code changes). No API key needed for this download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src/ ./src/
COPY data/eval/ ./data/eval/
COPY evals/ ./evals/

# Drop privileges.
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "deepdive.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
