
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose port (default FastAPI port)
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
