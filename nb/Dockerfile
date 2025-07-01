# Use the official Jupyter base image to ensure a working notebook environment
FROM jupyter/base-notebook

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies as root
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directory and set ownership for the jovyan user
RUN mkdir -p /usr/src/app/data && \
    chown -R $NB_UID:$NB_GID /usr/src/app/data

# Switch back to the standard notebook user
USER $NB_UID

# Download original sample data for compatibility
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv -o /usr/src/app/data/ground_truth_sample.csv

# Install Python dependencies using pip for stability
COPY --chown=$NB_UID:$NB_GID requirements.txt .
COPY --chown=$NB_UID:$NB_GID ../shared/requirements.txt shared-requirements.txt
RUN pip install --no-cache-dir -r shared-requirements.txt -r requirements.txt

# Copy shared modules and project files
COPY --chown=$NB_UID:$NB_GID ../shared/ /usr/src/shared/
COPY --chown=$NB_UID:$NB_GID . .
RUN pip install -e .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Expose the Jupyter Notebook port and set default command
EXPOSE 8888
ENV JUPYTER_ENABLE_LAB=yes
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]