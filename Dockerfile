FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    COAL_KB_CONFIG=configs/prod.yaml \
    COAL_KB_PUBLIC_MODE=true \
    COAL_KB_DATA_ROOT=/app/data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[docs]" \
    && mkdir -p /app/data

VOLUME ["/app/data"]
EXPOSE 8000

CMD ["python", "scripts/serve.py"]
