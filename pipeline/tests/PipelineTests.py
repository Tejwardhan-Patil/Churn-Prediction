import unittest
import json
import os
import subprocess
from airflow.models import DagBag
from luigi import Task, LocalTarget, build
from argo.workflows.client import V1alpha1Workflow, V1alpha1WorkflowSpec, WorkflowClient

# DataPipelineTask definition for testing
class DataPipelineTask(Task):
    """Luigi task for data preprocessing"""
    
    def output(self):
        return LocalTarget("/pipeline/output.csv")
    
    def run(self):
        with self.output().open('w') as out_file:
            out_file.write("Data output")

class PipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.airflow_dag_path = "/pipeline/airflow/dags"
        cls.luigi_task_path = "/pipeline/luigi/tasks"
        cls.argo_workflow_path = "/pipeline/argo_workflows"
        cls.airflow_config = "/pipeline/config/airflow_config.json"
        cls.argo_workflow_name = "churn-prediction-workflow"

    def test_airflow_dag_import(self):
        """Test if DAGs can be imported properly in Airflow."""
        dag_bag = DagBag(self.airflow_dag_path)
        self.assertFalse(
            len(dag_bag.import_errors) > 0, 
            f"Errors found in Airflow DAGs: {dag_bag.import_errors}"
        )

    def test_airflow_dag_schedule(self):
        """Test if the Airflow DAG is correctly scheduled."""
        dag_bag = DagBag(self.airflow_dag_path)
        dag = dag_bag.get_dag('churn_prediction_dag')
        self.assertIsNotNone(dag)
        self.assertEqual(dag.schedule_interval, '@daily', "DAG not scheduled as expected")

    def test_airflow_dag_run(self):
        """Test a manual run of the Airflow DAG."""
        result = subprocess.run(
            ["airflow", "dags", "trigger", "churn_prediction_dag"],
            capture_output=True,
            text=True
        )
        self.assertIn("Triggered", result.stdout, "Failed to trigger the Airflow DAG")

    def test_airflow_config(self):
        """Validate the Airflow config JSON file."""
        with open(self.airflow_config, 'r') as f:
            config_data = json.load(f)
        self.assertIn('schedule_interval', config_data, "Missing 'schedule_interval' in Airflow config")
        self.assertIn('default_args', config_data, "Missing 'default_args' in Airflow config")
        self.assertIsInstance(config_data['default_args'], dict, "'default_args' should be a dictionary")

    def test_luigi_task_completion(self):
        """Test if the Luigi task runs and completes successfully."""
        luigi_result = build([DataPipelineTask()], local_scheduler=True)
        self.assertTrue(luigi_result, "Luigi task failed to complete")

    def test_luigi_task_output(self):
        """Test the output of the Luigi task."""
        task = DataPipelineTask()
        output = task.output()
        self.assertTrue(output.exists(), "Expected output not found for Luigi task")

    def test_argo_workflow_creation(self):
        """Test if the Argo workflow is created and runs successfully."""
        client = WorkflowClient()
        workflow = V1alpha1Workflow(
            metadata={'generate_name': self.argo_workflow_name},
            spec=V1alpha1WorkflowSpec()
        )
        created_workflow = client.create_workflow(workflow)
        self.assertIsNotNone(created_workflow, "Failed to create Argo workflow")

    def test_argo_workflow_status(self):
        """Test the status of the Argo workflow."""
        client = WorkflowClient()
        workflow_status = client.get_workflow_status(self.argo_workflow_name)
        self.assertEqual(workflow_status, "Succeeded", "Argo workflow did not succeed")

    def test_argo_workflow_log_output(self):
        """Check logs from the Argo workflow to ensure no errors."""
        client = WorkflowClient()
        logs = client.get_workflow_logs(self.argo_workflow_name)
        self.assertNotIn("ERROR", logs, "Errors found in Argo workflow logs")

    def test_pipeline_execution_order(self):
        """Test the overall order of pipeline execution."""
        airflow_result = subprocess.run(
            ["airflow", "dags", "trigger", "churn_prediction_dag"],
            capture_output=True,
            text=True
        )
        luigi_result = build([DataPipelineTask()], local_scheduler=True)
        client = WorkflowClient()
        workflow_status = client.get_workflow_status(self.argo_workflow_name)

        self.assertIn("Triggered", airflow_result.stdout, "Failed to trigger Airflow DAG")
        self.assertTrue(luigi_result, "Luigi task failed")
        self.assertEqual(workflow_status, "Succeeded", "Argo workflow failed to complete")

    def test_airflow_dag_integrity(self):
        """Test if the DAGs maintain structural integrity after multiple runs."""
        dag_bag = DagBag(self.airflow_dag_path)
        dag = dag_bag.get_dag('churn_prediction_dag')
        runs_before = len(dag.get_task_instances())
        result = subprocess.run(
            ["airflow", "dags", "trigger", "churn_prediction_dag"],
            capture_output=True,
            text=True
        )
        runs_after = len(dag.get_task_instances())
        self.assertGreater(runs_after, runs_before, "DAG run did not increase task instances")

    def test_luigi_task_resilience(self):
        """Test the Luigi task for resilience against errors."""
        try:
            build([DataPipelineTask()], local_scheduler=True)
        except Exception as e:
            self.fail(f"Luigi task failed unexpectedly: {e}")

    def test_argo_workflow_resilience(self):
        """Test the Argo workflow for resilience against errors."""
        client = WorkflowClient()
        try:
            workflow = V1alpha1Workflow(
                metadata={'generate_name': self.argo_workflow_name},
                spec=V1alpha1WorkflowSpec()
            )
            client.create_workflow(workflow)
        except Exception as e:
            self.fail(f"Argo workflow failed unexpectedly: {e}")

    def test_pipeline_logs_monitoring(self):
        """Test if logs from pipeline components are being monitored correctly."""
        airflow_logs = subprocess.run(
            ["airflow", "logs", "churn_prediction_dag"],
            capture_output=True,
            text=True
        )
        luigi_logs = subprocess.run(
            ["luigi", "logs"],
            capture_output=True,
            text=True
        )
        client = WorkflowClient()
        argo_logs = client.get_workflow_logs(self.argo_workflow_name)

        self.assertNotIn("ERROR", airflow_logs.stdout, "Errors found in Airflow logs")
        self.assertNotIn("ERROR", luigi_logs.stdout, "Errors found in Luigi logs")
        self.assertNotIn("ERROR", argo_logs, "Errors found in Argo workflow logs")

if __name__ == '__main__':
    unittest.main()