# Base image with Python 3.9 and R
FROM rocker/r-ver:4.1.0

# Install Python 3.9 and required packages
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    build-essential \
    && apt-get clean

# Set up a working directory
WORKDIR /app

# Copy the API files into the container
COPY deployment/api /app/api
COPY deployment/scripts /app/scripts
COPY models /app/models
COPY data /app/data

# Install R packages from packages.R
COPY deployment/api/packages.R /app/packages.R
RUN Rscript /app/packages.R

# Install Python packages from requirements.txt
COPY deployment/api/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Expose the Flask (or FastAPI) port for Python API
EXPOSE 5000

# Expose the R Plumber API port
EXPOSE 8000

# Run both Python and R-based API in the same container
CMD ["sh", "-c", "python3 /app/api/app.py & Rscript -e 'pr <- plumber::plumb(\"/app/api/app.R\"); pr$run(host=\"0.0.0.0\", port=8000)'"]