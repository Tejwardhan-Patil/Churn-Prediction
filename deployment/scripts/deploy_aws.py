import boto3
import os
import time
import json
from botocore.exceptions import NoCredentialsError

# AWS client initialization
s3 = boto3.client('s3')
ec2 = boto3.resource('ec2')
lambda_client = boto3.client('lambda')

# Configuration variables
MODEL_LOCAL_PATH = "/mnt/data/models/best_model.pkl"
S3_BUCKET_NAME = "ml-model-deploy-bucket"
S3_MODEL_KEY = "models/best_model.pkl"
AWS_REGION = "us-east-1"
EC2_INSTANCE_TYPE = "t2.micro"
EC2_KEY_PAIR_NAME = "aws-deploy-key"
EC2_SECURITY_GROUP = "deploy-sg"
LAMBDA_FUNCTION_NAME = "ChurnPredictionLambda"

# Upload model to S3
def upload_model_to_s3():
    try:
        s3.upload_file(MODEL_LOCAL_PATH, S3_BUCKET_NAME, S3_MODEL_KEY)
        print("Model successfully uploaded to S3.")
    except FileNotFoundError:
        print("The model file was not found.")
    except NoCredentialsError:
        print("AWS credentials not available.")
        
# Create EC2 instance for deployment
def create_ec2_instance():
    try:
        instances = ec2.create_instances(
            ImageId='ami-12345678', 
            MinCount=1,
            MaxCount=1,
            InstanceType=EC2_INSTANCE_TYPE,
            KeyName=EC2_KEY_PAIR_NAME,
            SecurityGroups=[EC2_SECURITY_GROUP]
        )
        instance_id = instances[0].id
        print(f"EC2 instance {instance_id} created. Waiting for it to run...")
        instances[0].wait_until_running()
        instance = ec2.Instance(instance_id)
        print(f"EC2 instance {instance_id} running with public DNS: {instance.public_dns_name}")
        return instance_id
    except Exception as e:
        print(f"Error creating EC2 instance: {str(e)}")

# Deploy Lambda function for inference
def deploy_lambda_function():
    with open("lambda_deployment_package.zip", "rb") as f:
        zipped_code = f.read()
    
    try:
        response = lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime='python3.8',
            Role='arn:aws:iam::123456789012:role/execution_role',  # IAM role for Lambda
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zipped_code},
            Timeout=300,
            MemorySize=128,
            Publish=True
        )
        print(f"Lambda function {LAMBDA_FUNCTION_NAME} created.")
        return response
    except Exception as e:
        print(f"Error creating Lambda function: {str(e)}")

# Configure security group for EC2
def create_security_group():
    try:
        response = ec2.create_security_group(
            GroupName=EC2_SECURITY_GROUP,
            Description='Security group for ML model deployment'
        )
        security_group_id = response['GroupId']
        ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]  # SSH access
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]  # HTTP access
                }
            ]
        )
        print(f"Security group {EC2_SECURITY_GROUP} created with ID {security_group_id}.")
    except Exception as e:
        print(f"Error creating security group: {str(e)}")

# Configure AWS Lambda function environment
def configure_lambda_environment():
    try:
        response = lambda_client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Environment={
                'Variables': {
                    'S3_BUCKET': S3_BUCKET_NAME,
                    'MODEL_KEY': S3_MODEL_KEY
                }
            }
        )
        print("Lambda environment variables configured.")
        return response
    except Exception as e:
        print(f"Error configuring Lambda environment: {str(e)}")

# Function to deploy model to EC2
def deploy_model_to_ec2(instance_id):
    instance = ec2.Instance(instance_id)
    instance_dns = instance.public_dns_name
    
    # SCP the model to the EC2 instance
    try:
        os.system(f"scp -i {EC2_KEY_PAIR_NAME}.pem {MODEL_LOCAL_PATH} ec2-user@{instance_dns}:/home/ec2-user/")
        print(f"Model deployed to EC2 instance at {instance_dns}.")
    except Exception as e:
        print(f"Error deploying model to EC2: {str(e)}")

# Create IAM role for Lambda
def create_iam_role_for_lambda():
    iam_client = boto3.client('iam')
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        role = iam_client.create_role(
            RoleName="LambdaExecutionRole",
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description="Role to allow Lambda to invoke other AWS services."
        )
        print(f"IAM role created: {role['Role']['Arn']}")
        return role['Role']['Arn']
    except Exception as e:
        print(f"Error creating IAM role: {str(e)}")

# Main deployment function
def deploy_model_to_aws():
    upload_model_to_s3()
    
    # Create security group
    create_security_group()
    
    # Create EC2 instance
    ec2_instance_id = create_ec2_instance()
    
    if ec2_instance_id:
        # Deploy model to EC2
        deploy_model_to_ec2(ec2_instance_id)
    
    # Deploy Lambda function
    deploy_lambda_function()
    
    # Configure Lambda environment
    configure_lambda_environment()

if __name__ == "__main__":
    deploy_model_to_aws()