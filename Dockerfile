# ─────────────────────────────────────────────────────────────────────────────
# RepoContext AI — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t repo-context-ai .
# Run:    docker compose up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim AS base

# Keep Python output unbuffered for container log streaming.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install only the OS-level packages needed by chromadb and onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies layer (cached as long as requirements.txt is unchanged)
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Application layer
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS app

# Copy source files — .dockerignore excludes venv, data, .git, etc.
COPY agent/    ./agent/
COPY api/      ./api/
COPY core/     ./core/
COPY evaluation/ ./evaluation/
COPY app.py    ./app.py
COPY rag.py    ./rag.py

# Create the data directories that the app writes to at runtime.
RUN mkdir -p data/chroma data/repositories

# Expose the default API port.
EXPOSE 8000

# Use a non-root user for security.
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check using the liveness endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch with uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
