import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
import joblib

def load_model(model_path):
    """Load the model from the given file path."""
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """Evaluate the given model using test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print a detailed classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Create and plot confusion matrix heatmap
    plot_confusion_matrix(cm)

    # Return the metrics as a dictionary
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc
    }

def plot_confusion_matrix(cm):
    """Plot the confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """Plot the ROC curve."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label="ROC curve (AUC = %0.2f)" % roc_auc_score(y_test, y_prob))
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def save_evaluation_results(results, output_file):
    """Save evaluation metrics to a CSV file."""
    df = pd.DataFrame([results])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def run_evaluation_pipeline(model_path, test_data_path, results_output_path):
    """Run the entire model evaluation pipeline."""
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']

    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    results = evaluate_model(model, X_test, y_test)

    # Plot the ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_prob)

    # Save evaluation results
    save_evaluation_results(results, results_output_path)

if __name__ == "__main__":
    # Define paths for model and test data
    model_path = 'models/saved_models/best_model.pkl'
    test_data_path = 'data/processed/test_data.csv'
    results_output_path = 'experiments/results/evaluation_results.csv'

    # Run the evaluation pipeline
    run_evaluation_pipeline(model_path, test_data_path, results_output_path)