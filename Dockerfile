FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY . .

# Re-install in editable mode with the full source
RUN pip install --no-cache-dir -e .

# Pre-download the embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Expose ports: 8000 for API, 8501 for Streamlit
EXPOSE 8000 8501

# Default: run the API server
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
