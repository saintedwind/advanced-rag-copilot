FROM python:3.11-slim as base

WORKDIR /app

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends     curl     build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
