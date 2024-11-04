#!/bin/bash

# Exit script on any error
set -e

# Define directories
PROJECT_ROOT=$(dirname "$(realpath "$0")")/..
PYTHON_API_DIR="$PROJECT_ROOT/deployment/api"
R_API_DIR="$PROJECT_ROOT/deployment/api"
DOCKER_DIR="$PROJECT_ROOT/deployment/docker"
AIRFLOW_DAG_DIR="$PROJECT_ROOT/pipeline/airflow/dags"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r "$PYTHON_API_DIR/requirements.txt"

# Install R dependencies
echo "Installing R dependencies..."
Rscript -e "install.packages('plumber')"
Rscript -e "source('$R_API_DIR/packages.R')"

# Build Docker image for the API
echo "Building Docker image..."
docker build -t churn-api "$DOCKER_DIR"

# Initialize Airflow environment
echo "Initializing Airflow environment..."
if [ -d "$AIRFLOW_DAG_DIR" ]; then
    airflow db init
    airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@website.com
fi

# Run any database migrations
echo "Running database migrations..."
"$PROJECT_ROOT/scripts/migrate_db.sh"

# Run tests
echo "Running unit tests..."
pytest "$PROJECT_ROOT/tests/unit_tests/"

echo "Running integration tests..."
pytest "$PROJECT_ROOT/tests/integration_tests/"

echo "Running R tests..."
Rscript "$PROJECT_ROOT/models/tests/test_models.R"

# Lint Python code
echo "Linting Python code..."
flake8 "$PROJECT_ROOT"

# Lint R code
echo "Linting R code..."
Rscript -e "lintr::lint_dir('$PROJECT_ROOT')"

echo "Build process completed successfully."