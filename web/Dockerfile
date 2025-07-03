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
COPY shared/requirements.txt shared-requirements.txt
RUN pip install --no-cache-dir -r shared-requirements.txt -r requirements.txt

# Copy shared modules and application source code
COPY shared/ /app/shared/
COPY web/src/ /app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application - make init.sh executable first
CMD ["sh", "-c", "chmod +x /app/init.sh && /app/init.sh"]