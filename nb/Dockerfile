# Use the official Jupyter base image to ensure a working notebook environment
FROM jupyter/base-notebook

# Set the working directory
WORKDIR /usr/src/app

# --- Install uv (a faster pip) using a multi-stage build ---
# This copies the pre-built uv binary from the official image, which is the most reliable method.
COPY --from=ghcr.io/astral-sh/uv:0.7.17 /uv /usr/local/bin/

# --- System Dependencies ---
# Install system-level dependencies as root
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies & Project Installation (as root) ---
# 1. Copy requirements and install using uv.
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# 2. Copy the rest of the application code.
COPY . .

# 3. Install the project in editable mode using uv.
RUN uv pip install --system --no-cache -e .

# --- Post-Install Setup & Final User Switch ---
# Create a data directory and set correct ownership before switching user
RUN mkdir -p /usr/src/app/data && \
    chown -R $NB_UID:$NB_GID /usr/src/app/data

# Switch to the final user
USER $NB_UID

# Download the original ground truth sample for compatibility
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv -o /usr/src/app/data/ground_truth_sample.csv

# Download NLTK data required by the ingredient-parser-nlp library
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Expose the Jupyter Notebook port and set default command
EXPOSE 8888
ENV JUPYTER_ENABLE_LAB=yes
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]