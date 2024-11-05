import pandas as pd

# Load your CSV file (update the path if needed)
file_path = '/Users/vishwanathgowda/Documents/SecureSense/Data set/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
data = pd.read_csv(file_path)

# Print the first few rows of the dataframe
print(data.head())

# Print the column names
print("Column Names:")
print(data.columns)
