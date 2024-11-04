#!/bin/bash

# Exit script on any error
set -e

# Constants
DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
DEPLOYMENT_TARGET=""

# Functions
usage() {
    echo "Usage: $0 [aws|gcp]"
    echo "Deploy the system to the specified cloud provider."
    echo "  aws   Deploy to AWS using deployment/scripts/deploy_aws.py"
    echo "  gcp   Deploy to GCP using deployment/scripts/deploy_gcp.py"
    exit 1
}

# Parse deployment target
if [ $# -ne 1 ]; then
    usage
else
    DEPLOYMENT_TARGET=$1
fi

# Build Docker image
echo "Building Docker image..."
docker-compose -f $DOCKER_COMPOSE_FILE build

# Push Docker image to container registry
if [ "$DEPLOYMENT_TARGET" == "aws" ]; then
    echo "Deploying to AWS..."
    python3 deployment/scripts/deploy_aws.py
elif [ "$DEPLOYMENT_TARGET" == "gcp" ]; then
    echo "Deploying to GCP..."
    python3 deployment/scripts/deploy_gcp.py
else
    usage
fi

# Run Docker containers
echo "Running containers in the background..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d

# Verify deployment success
echo "Deployment complete. Checking services status..."
docker-compose -f $DOCKER_COMPOSE_FILE ps