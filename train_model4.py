import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Set file path
file_path = '/Users/vishwanathgowda/Documents/SecureSense/Monday-WorkingHours.pcap_ISCX.csv'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load dataset
    df = pd.read_csv(file_path)

    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Split features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Replace infinite values and fill NaNs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # Check and cap any remaining infinities to a large value
    X = np.where(np.isinf(X), 1e12, X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    classifier = RandomForestClassifier(random_state=42)

    # Fit the model
    classifier.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = classifier.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
