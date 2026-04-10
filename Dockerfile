# ---------------------------------------------------------------
# Dockerfile — Energy Intelligence Platform (FastAPI)
# ---------------------------------------------------------------
# Build:  docker build -t energy-intelligence-platform .
# Run:    docker run -p 8000:8000 \
#           -v $(pwd)/data:/app/data \
#           --env-file .env \
#           energy-intelligence-platform
# ---------------------------------------------------------------

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/     ./api/
COPY src/     ./src/
COPY models/  ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]