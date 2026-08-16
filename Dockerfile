FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY benchmark ./benchmark
COPY corpus.lock.json ./corpus.lock.json
RUN pip install .
EXPOSE 8000
CMD ["uvicorn", "kube_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
