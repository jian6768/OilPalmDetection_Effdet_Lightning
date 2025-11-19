FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Avoid writing .pyc files, unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app
COPY weights/ ./weights

# Cloud Run expects the container to listen on $PORT (default 8080)
ENV PORT=8080

# Expose the port (use 8080 for Cloud Run)
EXPOSE 8080

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
