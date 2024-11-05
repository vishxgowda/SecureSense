import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Read the dataset
df = pd.read_csv('/Users/vishwanathgowda/Documents/SecureSense/Data set/Friday-WorkingHours-Morning.pcap_ISCX.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print out the columns to check the exact names
print("Columns in the DataFrame:", df.columns.tolist())

# Clean the dataset: replace infinite values and drop NaN rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Check the target column name and set it accordingly
target_column = 'Label'  # Update this if necessary after checking the printed column names

# Verify if the target column exists
if target_column in df.columns:
    y = df[target_column]  # Adjust if necessary
else:
    raise KeyError(f"Target column '{target_column}' does not exist in the DataFrame.")

X = df.drop(target_column, axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
