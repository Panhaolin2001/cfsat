import os
import pandas as pd

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Define input and output paths
input = os.path.join(project_root, 'output', 'Phase1_SynerPairLists.csv')
output = os.path.join(project_root, 'output', 'Phase1_Filtered_SynerPairLists.csv')

# Read the CSV file
df = pd.read_csv(input)

# Remove rows where the Synerpairlist column is an empty list
df = df[df['Synerpairlist'].apply(lambda x: x != '[]')]

# Save the processed CSV file
df.to_csv(output, index=False)

print("Rows with empty lists have been successfully removed and saved to Phase1_Filtered_SynerPairLists.csv")
