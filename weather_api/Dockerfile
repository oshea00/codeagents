# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies and pytest
RUN pip install --no-cache-dir -r requirements.txt pytest

# Copy all Python files from source directory
COPY src /app/

# Default command to run tests
CMD ["python", "-m", "pytest"]
