FROM python:3.12-slim

WORKDIR /app

# System deps for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install in layers for better caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi uvicorn pydantic httpx python-dotenv \
    sentence-transformers faiss-cpu numpy \
    openai tiktoken \
    pytest

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
