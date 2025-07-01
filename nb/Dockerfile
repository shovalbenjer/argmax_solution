# Use the official Jupyter base image to ensure a working notebook environment
FROM jupyter/base-notebook

# Set the working directory
WORKDIR /usr/src/app

# --- System Dependencies ---
# Switch to root user to install system-level dependencies
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
USER $NB_UID

# --- Python Dependencies (Optimized for Caching) ---
# 1. Copy only the requirements file first.
# This layer will be cached as long as requirements.txt does not change.
COPY requirements.txt .

# 2. Install dependencies. This is the most time-consuming step.
# It will only re-run if requirements.txt is modified.
RUN pip install --no-cache-dir -r requirements.txt

# --- Project Installation ---
# 3. Copy the rest of the application code.
# Changes to your source code will now only invalidate this layer and subsequent ones.
COPY . .

# 4. Install the project in editable mode.
RUN pip install -e .

# --- Post-Install Setup ---
# Download NLTK data required by the ingredient-parser-nlp library
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Expose the Jupyter Notebook port and set default command
EXPOSE 8888
ENV JUPYTER_ENABLE_LAB=yes
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]