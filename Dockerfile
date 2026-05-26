# Use a slim Python 3.13 image
FROM python:3.13-slim

# Set environment variables to ensure optimal Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages (like xgboost/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server via Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
