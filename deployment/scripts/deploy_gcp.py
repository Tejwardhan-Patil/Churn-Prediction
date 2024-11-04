import os
import subprocess
from google.cloud import storage
from googleapiclient.discovery import build
import google.auth
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Authenticate with Google Cloud
def authenticate_gcp():
    credentials, project = google.auth.default()
    return credentials, project

# Upload model to Google Cloud Storage
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    logger.info(f"Model {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}")

# Deploy model to AI Platform
def deploy_model_ai_platform(model_name, model_version, model_path, project):
    ml_client = build('ml', 'v1', credentials=google.auth.default()[0])
    
    model_body = {
        'name': model_name,
        'regions': ['us-central1']
    }
    request = ml_client.projects().models().create(parent=f'projects/{project}', body=model_body)
    request.execute()
    
    version_body = {
        'name': model_version,
        'deploymentUri': model_path,
        'runtimeVersion': '2.3',
        'framework': 'TENSORFLOW',
        'pythonVersion': '3.7'
    }
    request = ml_client.projects().models().versions().create(
        parent=f'projects/{project}/models/{model_name}', body=version_body)
    request.execute()
    logger.info(f"Model {model_name} version {model_version} deployed to AI Platform")

# Build and push Docker image to Google Container Registry (GCR)
def build_and_push_docker_image(image_name, gcr_path, dockerfile_dir):
    logger.info(f"Building Docker image {image_name}")
    subprocess.run(['docker', 'build', '-t', image_name, dockerfile_dir], check=True)
    
    logger.info(f"Pushing image {image_name} to GCR: {gcr_path}")
    subprocess.run(['docker', 'tag', image_name, gcr_path], check=True)
    subprocess.run(['docker', 'push', gcr_path], check=True)

# Deploy model on Kubernetes Engine (GKE)
def deploy_to_gke(cluster_name, zone, deployment_file):
    logger.info(f"Connecting to GKE cluster {cluster_name}")
    subprocess.run(['gcloud', 'container', 'clusters', 'get-credentials', cluster_name, '--zone', zone], check=True)
    
    logger.info("Deploying application to GKE")
    subprocess.run(['kubectl', 'apply', '-f', deployment_file], check=True)

# Set up Cloud Run service
def deploy_to_cloud_run(image_uri, service_name, project, region='us-central1'):
    logger.info(f"Deploying to Cloud Run service {service_name}")
    subprocess.run(['gcloud', 'run', 'deploy', service_name, '--image', image_uri, '--platform', 'managed', '--region', region, '--allow-unauthenticated'], check=True)
    logger.info(f"Cloud Run service {service_name} deployed successfully")

# Setup Pub/Sub for triggers
def setup_pubsub(topic_name):
    logger.info(f"Setting up Pub/Sub topic: {topic_name}")
    subprocess.run(['gcloud', 'pubsub', 'topics', 'create', topic_name], check=True)
    logger.info(f"Pub/Sub topic {topic_name} created successfully")

# Set up Cloud Functions for API endpoint
def deploy_cloud_function(function_name, source_path, entry_point, runtime='python310'):
    logger.info(f"Deploying Cloud Function {function_name}")
    subprocess.run([
        'gcloud', 'functions', 'deploy', function_name,
        '--runtime', runtime,
        '--trigger-http',
        '--allow-unauthenticated',
        '--source', source_path,
        '--entry-point', entry_point], check=True)
    logger.info(f"Cloud Function {function_name} deployed successfully")

# Deploy model workflow
def deploy_model_workflow(config_file):
    # Load configuration
    config = load_config(config_file)
    
    # Authenticate and set project
    credentials, project = authenticate_gcp()
    
    # Upload model to GCS
    model_local_path = config['model']['local_path']
    gcs_bucket = config['gcs']['bucket_name']
    gcs_blob_name = config['gcs']['blob_name']
    upload_to_gcs(gcs_bucket, model_local_path, gcs_blob_name)
    
    # Deploy to AI Platform
    model_name = config['ai_platform']['model_name']
    model_version = config['ai_platform']['model_version']
    model_gcs_path = f"gs://{gcs_bucket}/{gcs_blob_name}"
    deploy_model_ai_platform(model_name, model_version, model_gcs_path, project)
    
    # Build and push Docker image to GCR
    docker_image_name = config['gcr']['image_name']
    gcr_path = f"gcr.io/{project}/{docker_image_name}"
    dockerfile_dir = config['gcr']['dockerfile_dir']
    build_and_push_docker_image(docker_image_name, gcr_path, dockerfile_dir)
    
    # Deploy to GKE
    cluster_name = config['gke']['cluster_name']
    zone = config['gke']['zone']
    gke_deployment_file = config['gke']['deployment_file']
    deploy_to_gke(cluster_name, zone, gke_deployment_file)
    
    # Deploy to Cloud Run
    cloud_run_service = config['cloud_run']['service_name']
    cloud_run_image_uri = f"gcr.io/{project}/{docker_image_name}"
    region = config['cloud_run']['region']
    deploy_to_cloud_run(cloud_run_image_uri, cloud_run_service, project, region)
    
    # Set up Pub/Sub
    pubsub_topic = config['pubsub']['topic_name']
    setup_pubsub(pubsub_topic)
    
    # Deploy Cloud Function for API
    cloud_function_name = config['cloud_function']['function_name']
    cloud_function_source = config['cloud_function']['source_path']
    cloud_function_entry = config['cloud_function']['entry_point']
    deploy_cloud_function(cloud_function_name, cloud_function_source, cloud_function_entry)
    
    logger.info("Model deployment workflow completed successfully.")

if __name__ == '__main__':
    deploy_model_workflow('config.yaml')