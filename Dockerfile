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

# Using Supervisord to run FastAPI and Streamlit in parallel
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY api/        ./api/
COPY src/        ./src/
COPY dashboard/  ./dashboard/
COPY models/     ./models/
COPY data/       ./data/
COPY main.py     ./main.py

# PYTHONPATH so that api/main.py can import from src/
ENV PYTHONPATH=/app

# Config supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]