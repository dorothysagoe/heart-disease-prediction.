#The following code is applied for data processing

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the heart disease dataset.
    Args:
        filepath (str): Path to the heart.csv file.
    Returns:
        X_train, X_test, y_train, y_test: Split and scaled data.
        preprocessor: The fitted preprocessing pipeline for use on new data.
    """
    # Load the data
    df = pd.read_csv(filepath)

    # Display basic info
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Define features and target
    # Note: Adjust these column names based on your actual dataset!
    # This is a common structure for the Cleveland heart disease dataset.
    X = df.drop('target', axis=1) # 'target' is typically the label (0 = no disease, 1 = disease)
    y = df['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# For testing the script directly
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('../../data/heart.csv')
