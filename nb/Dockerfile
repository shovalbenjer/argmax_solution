FROM python:3.11-slim

WORKDIR /app

# Install system dependencies first, as they change infrequently
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Optimized Caching for Python Dependencies ---
# 1. Copy only the requirements file first.
# As long as this file doesn't change, Docker will use the cached layer for pip install.
COPY nb/requirements.txt .

# 2. Install dependencies. This is the most time-consuming layer.
RUN pip install --no-cache-dir -r requirements.txt

# 3. Now, copy the rest of the application code.
# Changes to your source code will now only invalidate this layer and subsequent ones.
COPY nb/src /app/src
COPY tests /app/tests
COPY nb/setup.py /app/setup.py

# --- Application Setup ---
# Create data directory (can be done here, as it's a fast operation)
RUN mkdir -p /app/data

# Download NLTK data required by the parser
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Expose the port
EXPOSE 8888

# Set the entrypoint to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app", "--ServerApp.token=''", "--ServerApp.password=''"]