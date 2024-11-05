import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the dataset
file_path = '/Users/vishwanathgowda/Documents/SecureSense/Data set/Friday-WorkingHours-Morning.pcap_ISCX.csv'
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Specify the target column
target_column = 'Label'

# Check if the target column exists
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found in dataset. Please verify the column name.")

# Separate features and target variable
X = df.drop(target_column, axis=1)
y = df[target_column]

# Replace infinite values with NaN, then drop rows with NaN values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Align y with the cleaned X to remove corresponding rows
y = y[X.index]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
