FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for camelot-py, chromadb, and other packages
RUN apt-get update && apt-get install -y \
    ghostscript \
    libpoppler-dev \
    poppler-utils \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY *.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]