# ALLM API — deployable core (Roadmap M52).
#
# A minimal image that serves the HTTP boundary. The heavy ML extras are
# NOT installed: everything in src/allm imports them lazily, so the API,
# evidence loop, KEL, KDP, practice and events all run without torch.
FROM python:3.12-slim AS base

# No .pyc, unbuffered logs — standard for containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
# Copy only what the build backend needs to resolve metadata.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# The store lives on a mounted volume, never in the image.
ENV ALLM_STORAGE__PATH=/data/allm.sqlite3
RUN mkdir -p /data && useradd --create-home --uid 10001 allm \
    && chown -R allm:allm /data /app
USER allm

EXPOSE 8000

# Liveness for orchestrators; readiness (/ready) is checked by compose.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

# create_default_app honours ALLM_STORAGE__PATH and ALLM_API_TOKEN.
CMD ["uvicorn", "--factory", "allm.api.app:create_default_app", \
     "--host", "0.0.0.0", "--port", "8000"]
