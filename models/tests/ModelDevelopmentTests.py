import unittest
from models.baseline_model import BaselineModel
from models.advanced_models import AdvancedModel
from models.hyperparameter_tuning import HyperparameterTuner
from models.model_evaluation import ModelEvaluator
from models.model_selection import ModelSelector
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelDevelopmentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the data and environment before running tests."""
        cls.X, cls.y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    def test_baseline_model_training(self):
        """Test training of the baseline model (Logistic Regression)."""
        model = BaselineModel()
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.6, "Baseline model accuracy is too low.")

    def test_advanced_model_training(self):
        """Test training of advanced models (Random Forest, XGBoost, Neural Networks)."""
        model = AdvancedModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.7, "Random Forest accuracy is too low.")
        
        model = AdvancedModel(model_type='xgboost')
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.75, "XGBoost accuracy is too low.")

        model = AdvancedModel(model_type='neural_network')
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.7, "Neural Network accuracy is too low.")

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning using GridSearch or Optuna."""
        tuner = HyperparameterTuner(model_type='random_forest')
        best_params = tuner.tune(self.X_train, self.y_train)
        self.assertIsNotNone(best_params, "Hyperparameter tuning failed to return best parameters.")
        
        model = AdvancedModel(model_type='random_forest', params=best_params)
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.75, "Tuned Random Forest accuracy is too low.")

    def test_model_evaluation(self):
        """Test evaluation metrics for the trained models."""
        model = BaselineModel()
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        evaluator = ModelEvaluator()
        accuracy = evaluator.evaluate(predictions, self.y_test, metric='accuracy')
        precision = evaluator.evaluate(predictions, self.y_test, metric='precision')
        recall = evaluator.evaluate(predictions, self.y_test, metric='recall')
        f1 = evaluator.evaluate(predictions, self.y_test, metric='f1')
        
        self.assertGreater(accuracy, 0.6, "Baseline model accuracy is too low.")
        self.assertGreater(precision, 0.6, "Baseline model precision is too low.")
        self.assertGreater(recall, 0.6, "Baseline model recall is too low.")
        self.assertGreater(f1, 0.6, "Baseline model F1-score is too low.")

    def test_model_selection(self):
        """Test model selection to choose the best model based on evaluation metrics."""
        selector = ModelSelector(models=['baseline', 'random_forest', 'xgboost'])
        best_model = selector.select_best(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertIsNotNone(best_model, "Model selection failed to select a best-performing model.")
        
        best_predictions = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, best_predictions)
        self.assertGreater(accuracy, 0.75, "Selected model accuracy is too low.")

    def test_invalid_model_type(self):
        """Test that an invalid model type raises an error."""
        with self.assertRaises(ValueError):
            AdvancedModel(model_type='invalid_model')

    def test_model_persistence(self):
        """Test saving and loading of models."""
        model = AdvancedModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        # Save model
        model.save_model('random_forest.pkl')
        
        # Load model
        loaded_model = AdvancedModel.load_model('random_forest.pkl')
        loaded_predictions = loaded_model.predict(self.X_test)
        loaded_accuracy = accuracy_score(self.y_test, loaded_predictions)
        
        self.assertGreater(loaded_accuracy, 0.7, "Loaded model accuracy is too low.")

    def test_cross_validation(self):
        """Test cross-validation during model training."""
        model = AdvancedModel(model_type='xgboost')
        scores = model.cross_validate(self.X, self.y, cv=5)
        self.assertGreaterEqual(min(scores), 0.7, "Cross-validation minimum score is too low.")
        self.assertLessEqual(max(scores), 1.0, "Cross-validation maximum score is invalid.")

if __name__ == '__main__':
    unittest.main()