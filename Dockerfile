# Use lightweight official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (build tools for numpy/scipy/xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Run training to ensure fresh model artifacts are compiled
RUN python -m src.train

# Expose ports: 8000 for FastAPI REST API, 8501 for Streamlit Dashboard
EXPOSE 8000 8501

# Default command launches the FastAPI microservice
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
