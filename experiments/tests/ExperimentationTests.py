import unittest
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

class ExperimentationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load configurations and data before running tests."""
        cls.config_path = os.path.join(os.path.dirname(__file__), '../configs/experiment_01.yaml')
        cls.data_path = os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_customer_data.csv')
        with open(cls.config_path, 'r') as file:
            cls.config = yaml.safe_load(file)
        cls.data = pd.read_csv(cls.data_path)
        
    def test_data_loading(self):
        """Test if the data is loaded correctly."""
        self.assertFalse(self.data.empty, "Data not loaded properly")
        self.assertTrue(len(self.data.columns) > 0, "No columns in data")

    def test_train_test_split(self):
        """Test if data splitting works as expected."""
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])
        self.assertEqual(len(X_train), int((1 - self.config['split_ratio']) * len(X)), "Train size incorrect")
        self.assertEqual(len(X_test), int(self.config['split_ratio'] * len(X)), "Test size incorrect")

    def test_model_training(self):
        """Test if the model training process is functioning."""
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])

        # Initialize model
        model = RandomForestClassifier(n_estimators=self.config['n_estimators'], random_state=self.config['random_state'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        self.assertGreater(accuracy, self.config['min_accuracy'], "Model accuracy is below the expected threshold")
        
        report = classification_report(y_test, predictions)
        print("\nClassification Report:\n", report)

    def test_hyperparameter_tuning(self):
        """Test if hyperparameter tuning is applied."""
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])

        model = RandomForestClassifier(random_state=self.config['random_state'])

        param_grid = {
            'n_estimators': self.config['tuning']['n_estimators'],
            'max_depth': self.config['tuning']['max_depth'],
            'min_samples_split': self.config['tuning']['min_samples_split']
        }
        
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print("\nBest Hyperparameters: ", best_params)
        
        self.assertIn(best_params['n_estimators'], self.config['tuning']['n_estimators'], "n_estimators tuning failed")
        self.assertIn(best_params['max_depth'], self.config['tuning']['max_depth'], "max_depth tuning failed")

    def test_model_saving(self):
        """Test if model is saved after training."""
        model = RandomForestClassifier(n_estimators=self.config['n_estimators'], random_state=self.config['random_state'])
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])
        model.fit(X_train, y_train)

        model_path = os.path.join(os.path.dirname(__file__), '../models/saved_models/best_model.pkl')
        joblib.dump(model, model_path)

        self.assertTrue(os.path.exists(model_path), "Model not saved properly")

    def test_model_loading(self):
        """Test if the saved model can be loaded and used."""
        model_path = os.path.join(os.path.dirname(__file__), '../models/saved_models/best_model.pkl')
        self.assertTrue(os.path.exists(model_path), "Model file not found")

        model = joblib.load(model_path)
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        self.assertGreater(accuracy, self.config['min_accuracy'], "Loaded model accuracy is below the threshold")
        
    def test_experiment_results_logging(self):
        """Test if experiment results are logged correctly."""
        X = self.data.drop(columns=['Churn'])
        y = self.data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])

        model = RandomForestClassifier(n_estimators=self.config['n_estimators'], random_state=self.config['random_state'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        result_path = os.path.join(os.path.dirname(__file__), '../experiments/results/experiment_results.csv')
        with open(result_path, 'w') as f:
            f.write("experiment_id,accuracy\n")
            f.write(f"exp_01,{accuracy}\n")

        self.assertTrue(os.path.exists(result_path), "Results file not created")
        
        results_df = pd.read_csv(result_path)
        self.assertEqual(results_df.shape[0], 1, "Results logging failed")
        self.assertAlmostEqual(results_df['accuracy'][0], accuracy, "Logged accuracy does not match expected value")

if __name__ == '__main__':
    unittest.main()