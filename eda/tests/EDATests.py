import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import pandas as pd
from eda.visualization.churn_distribution import visualize_churn_distribution 

class TestEDANotebooks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test data from the CSV files in the repository
        cls.customer_data = pd.read_csv('data/raw/customer_data.csv')
        cls.transaction_data = pd.read_csv('data/raw/transaction_data.csv')

    def load_and_run_notebook(self, notebook_path):
        """Helper function to load and run a Jupyter notebook."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        return nb

    def extract_variable(self, nb, var_name):
        """Executes the cells of a notebook and extracts a specific variable."""
        globals_dict = {}
        for cell in nb.cells:
            if cell.cell_type == 'code':
                try:
                    exec(cell.source, globals_dict)
                    if var_name in globals_dict:
                        return globals_dict[var_name]
                except Exception as e:
                    pass  # Skipping cells that raise exceptions (plotting)
        raise ValueError(f"Variable {var_name} not found in notebook.")

    def test_customer_eda_notebook(self):
        # Path to the customer EDA notebook
        notebook_path = 'eda/notebooks/eda_customer_data.ipynb'
        nb = self.load_and_run_notebook(notebook_path)

        # Extract and test summary statistics
        summary_stats = self.extract_variable(nb, 'summary_stats')
        self.assertIsInstance(summary_stats, pd.DataFrame)
        self.assertIn('age', summary_stats.columns)
        self.assertIn('mean', summary_stats.index)

    def test_transaction_eda_notebook(self):
        # Path to the transaction EDA notebook
        notebook_path = 'eda/notebooks/eda_transaction_data.ipynb'
        nb = self.load_and_run_notebook(notebook_path)

        # Extract and test transaction summary statistics
        summary_stats = self.extract_variable(nb, 'summary_stats')
        self.assertIsInstance(summary_stats, pd.DataFrame)
        self.assertIn('amount', summary_stats.columns)
        self.assertIn('sum', summary_stats.index)

    def test_visualization_churn_distribution(self):
        # Test churn distribution visualization 
        churn_distribution_plot = visualize_churn_distribution(self.customer_data)
        self.assertIsNotNone(churn_distribution_plot)

    def test_handling_invalid_data(self):
        # Path to customer EDA notebook
        notebook_path = 'eda/notebooks/eda_customer_data.ipynb'
        invalid_data = self.customer_data.copy()
        invalid_data['age'] = ['invalid', 45, 35, 50]  # Insert invalid data type

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            
            # Inject the invalid data into the notebook's execution context
            globals_dict = {'invalid_data': invalid_data}
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    exec(cell.source.replace('customer_data', 'invalid_data'), globals_dict)

            # Check if notebook raises ValueError for invalid data type
            with self.assertRaises(ValueError):
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})

    def test_outlier_detection_in_notebook(self):
        # Test outlier detection in customer EDA notebook
        notebook_path = 'eda/notebooks/eda_customer_data.ipynb'
        nb = self.load_and_run_notebook(notebook_path)

        # Extract and test outliers
        outliers = self.extract_variable(nb, 'outliers')
        self.assertIsInstance(outliers, pd.DataFrame)

    def test_plotting_transaction_trends(self):
        # Test transaction trend plotting in transaction EDA notebook
        notebook_path = 'eda/notebooks/eda_transaction_data.ipynb'
        nb = self.load_and_run_notebook(notebook_path)

        # Extract and test transaction trend plot
        trends_plot = self.extract_variable(nb, 'trends_plot')
        self.assertIsNotNone(trends_plot)

    def test_churn_by_age_visualization(self):
        # Test visualization of churn by age in customer EDA notebook
        notebook_path = 'eda/notebooks/eda_customer_data.ipynb'
        nb = self.load_and_run_notebook(notebook_path)

        # Extract and test churn by age plot
        age_churn_plot = self.extract_variable(nb, 'age_churn_plot')
        self.assertIsNotNone(age_churn_plot)

if __name__ == '__main__':
    unittest.main()