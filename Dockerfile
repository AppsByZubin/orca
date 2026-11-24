# ORCA staging/prod image. Single image, two roles (api | worker) chosen by the
# command - see deploy/compose.yaml.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ORCA_LOG_FORMAT=full

WORKDIR /app

# curl is only here for the healthcheck: spawning a Python interpreter per probe
# costs ~200ms+ and times out under load, which makes the container flap.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Dependency layer first so source edits do not reinstall the world.
COPY pyproject.toml README.md ./
COPY orca/__init__.py ./orca/__init__.py
RUN pip install --upgrade pip && pip install ".[parquet]"

COPY orca/ ./orca/
COPY config/ ./config/
COPY scripts/ ./scripts/
RUN pip install --no-deps --force-reinstall ".[parquet]"

# Writable, and owned by a non-root user: nothing here needs privileges.
RUN useradd --create-home --uid 10001 orca \
 && mkdir -p /data/models \
 && chown -R orca:orca /app /data
USER orca

# The installed package lives in site-packages, so ../config does not exist
# there: point CONFIG_DIR at the copies under /app explicitly.
ENV ORCA_CONFIG_DIR=/app/config \
    ORCA_DATA_DIR=/data \
    ORCA_DATABASE_URL=sqlite:////data/orca.db \
    ORCA_MODEL_DIR=/data/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS --max-time 8 http://127.0.0.1:8000/api/health > /dev/null || exit 1

CMD ["orca", "serve", "--host", "0.0.0.0", "--port", "8000"]
