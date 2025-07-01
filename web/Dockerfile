# Use a slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# --- Install uv (a faster pip) using a multi-stage build ---
COPY --from=ghcr.io/astral-sh/uv:0.7.17 /uv /usr/local/bin/

# --- System Dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies (Optimized for Caching) ---
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application source code
COPY src/ /app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]