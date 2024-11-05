import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np
import joblib  # Import joblib to save the model

# Define the directory containing your dataset
data_directory = '/Users/vishwanathgowda/Documents/SecureSense/Data set/'

# Initialize lists to store model results
results = []

# Loop through each CSV file in the specified directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_directory, filename)
        print(f'Processing file: {file_path}')

        # Load the dataset
        df = pd.read_csv(file_path)

        # Normalize column names by stripping whitespace
        df.columns = df.columns.str.strip()

        # Check if 'Label' is in the DataFrame
        if 'Label' not in df.columns:
            print(f"Error: 'Label' not found in columns.")
            continue

        # Display the columns and missing values
        print(f"Columns in the DataFrame: {df.columns.tolist()}")
        print(f"Missing values in the dataset:\n{df.isnull().sum()}")

        # Check for infinite values and replace them with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill or drop NaN values (here we drop any rows with NaN values)
        df.dropna(inplace=True)

        # Prepare the data
        X = df.drop(columns=['Label'])  # Features
        y = df['Label']  # Target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        report = classification_report(y_test, y_pred)
        results.append(report)

        print(f"Classification report for {filename}:\n{report}")

        # Save the trained model
        model_filename = os.path.join(data_directory, f'model_{filename}.joblib')
        joblib.dump(model, model_filename)
        print(f"Model saved as: {model_filename}")

# Display overall results
for i, report in enumerate(results):
    print(f"\nResults for file {i + 1}:\n{report}")
