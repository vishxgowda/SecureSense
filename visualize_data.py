import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/vishwanathgowda/Documents/SecureSense/Data set/CICIDS2017.csv')  # Make sure this path is correct

# Print column names to find the correct ones
print("Column Names:")
print(data.columns)

# Update the following line with the correct column names based on your dataset
# Example using 'Flow Duration' as an x-axis instead of 'Time'
plt.plot(data[' Flow Duration'], data[' Packet Length Mean'], label='Packet Length Mean')  # Adjust column names as needed
plt.xlabel('Flow Duration')
plt.ylabel('Packet Length Mean')
plt.title('Packet Length Over Flow Duration')
plt.legend()
plt.show()
