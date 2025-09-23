import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os
from .data_preprocessing import load_and_preprocess_data

def train_and_save_model(data_path, model_save_path='../models/best_model.pkl'):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform Grid Search with Cross-Validation
    print("\nStarting Grid Search...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix - Heart Disease Prediction')
    plt.savefig('../models/confusion_matrix.png') # Save the plot
    plt.show()

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model saved to {model_save_path}")

    return best_model, test_accuracy

if __name__ == "__main__":
    # Run the training pipeline
    model, accuracy = train_and_save_model('../../data/heart.csv')
