from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import os
import subprocess
from datetime import timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['alerts@website.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='A churn prediction pipeline using Python and R models',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False
)

# Define paths for scripts
BASE_DIR = '/usr/local/airflow/'
DATA_DIR = os.path.join(BASE_DIR, 'data/')
MODEL_DIR = os.path.join(BASE_DIR, 'models/')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts/')
FEATURES_DIR = os.path.join(BASE_DIR, 'features/')
DEPLOYMENT_DIR = os.path.join(BASE_DIR, 'deployment/')

# Function to run a script
def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    subprocess.run(['python', script_path], check=True)

# Task to preprocess data
def preprocess_data():
    script_path = os.path.join(SCRIPTS_DIR, 'preprocess.py')
    subprocess.run(['python', script_path], check=True)

# Task for feature engineering
def feature_engineering():
    script_path = os.path.join(SCRIPTS_DIR, 'feature_engineering.py')
    subprocess.run(['python', script_path], check=True)

# Task to split data
def data_split():
    script_path = os.path.join(SCRIPTS_DIR, 'data_split.py')
    subprocess.run(['python', script_path], check=True)

# Task for training models
def train_models():
    script_path = os.path.join(MODEL_DIR, 'baseline_model.py')
    subprocess.run(['python', script_path], check=True)

# Task to evaluate models
def evaluate_models():
    script_path = os.path.join(MODEL_DIR, 'model_evaluation.py')
    subprocess.run(['python', script_path], check=True)

# Task for model deployment
def deploy_model():
    script_path = os.path.join(DEPLOYMENT_DIR, 'deploy_aws.py')
    subprocess.run(['python', script_path], check=True)

# Preprocess task
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

# Feature engineering task
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag
)

# Data split task
data_split_task = PythonOperator(
    task_id='data_split',
    python_callable=data_split,
    dag=dag
)

# Model training task
train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

# Model evaluation task
evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag
)

# Model deployment task
deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# DAG dependencies
preprocess_task >> feature_engineering_task >> data_split_task >> train_models_task >> evaluate_models_task >> deploy_model_task

# Additional tasks for monitoring and reporting
def monitor_model_drift():
    script_path = os.path.join(DEPLOYMENT_DIR, 'monitor.py')
    subprocess.run(['python', script_path], check=True)

monitor_task = PythonOperator(
    task_id='monitor_model_drift',
    python_callable=monitor_model_drift,
    dag=dag
)

def alert_if_drift():
    script_path = os.path.join(DEPLOYMENT_DIR, 'alert.py')
    subprocess.run(['python', script_path], check=True)

alert_task = PythonOperator(
    task_id='alert_if_drift',
    python_callable=alert_if_drift,
    dag=dag
)

# Final dependencies for monitoring
evaluate_models_task >> monitor_task >> alert_task