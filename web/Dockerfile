FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p /app/data

# Download the parquet file
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/allrecipes.parquet -o /app/data/allrecipes.parquet

# Copy requirements first to leverage Docker cache
COPY web/requirements.txt .
# Use nb/setup.py instead of shared/requirements.txt
COPY nb/setup.py nb-setup.py
RUN pip install --no-cache-dir -r requirements.txt

# Copy modules and application source code - use nb/src instead of shared
COPY nb/src/ /app/nb_src/
COPY web/src/ /app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]