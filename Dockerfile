# ---------------------------------------------------------------
# Dockerfile — Energy Intelligence Platform (FastAPI)
# ---------------------------------------------------------------
# Builds a lightweight container that runs the FastAPI prediction API.
#
# Build:  docker build -t energy-intelligence-platform .
# Run:    docker run -p 8000:8000 energy-intelligence-platform
# ---------------------------------------------------------------

# Base image: official Python 3.11 slim (lightweight, no extras)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first (Docker caches this layer)
# If requirements.txt doesn't change, Docker won't reinstall packages
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY api/        ./api/
COPY src/        ./src/
COPY models/     ./models/

# Expose the port FastAPI will listen on
EXPOSE 8000

# Command to start the API when the container launches
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]