# FastAPI Backend for Label Studio Integration
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY label_api/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code (context is parent directory)
COPY label_api/ /app/label_api/
COPY utilities/ /app/utilities/
COPY config/ /app/config/

# Create data directories
RUN mkdir -p /app/data_labeling

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn - using websocket-compatible settings
CMD ["uvicorn", "label_api.legacy_main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--ws", "websockets", \
    "--ws-ping-interval", "20", \
    "--ws-ping-timeout", "20", \
    "--timeout-keep-alive", "65", \
    "--log-level", "info", \
    "--reload","--reload-dir","/app/label_api","--reload-dir","/app/utilities","--reload-dir","/app/data_labeling"]