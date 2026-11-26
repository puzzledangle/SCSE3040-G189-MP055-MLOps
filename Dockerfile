# Use official lightweight Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables for clean container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies first for efficient layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, models, configuration, and app package
COPY src/ /app/src/
COPY app/ /app/app/
COPY models/ /app/models/
COPY params.yaml /app/

# Expose standard container port
EXPOSE 8000

# Start Uvicorn ASGI server bound to 0.0.0.0
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
