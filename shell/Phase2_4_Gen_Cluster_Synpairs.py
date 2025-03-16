import os
import pandas as pd
import ast

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Define paths
synerpair_vec_path = os.path.join(project_root, 'output', 'Phase2_Synerpair_Vec.csv')
enumerated_pairs_path = os.path.join(project_root, 'output', 'Phase2_Enumerated_pairs.csv')
file_kmeans_path = os.path.join(project_root, 'output', 'Phase2_file_Kmeans100.csv')
output_path = os.path.join(project_root, 'output', 'Phase2_Cluster_Synerpairs.csv')

# Read data
synerpair_vec_df = pd.read_csv(synerpair_vec_path)
enumerated_pairs_df = pd.read_csv(enumerated_pairs_path)
file_kmeans_df = pd.read_csv(file_kmeans_path)

# Convert the synerpair column in the Enumerated_pairs.csv file to a dictionary
index_to_synerpair = dict(zip(enumerated_pairs_df['index'], enumerated_pairs_df['synerpair'].apply(ast.literal_eval)))

# Initialize an empty dictionary to store results
cluster_synerpairs = {}

# Iterate through file_kmeans_df to find the synerpair list for each Cluster Label
for _, row in file_kmeans_df.iterrows():
    filename = row['Filename']
    cluster_label = row['Cluster Label']
    
    # Find the pairVec corresponding to the filename
    pair_vec = synerpair_vec_df[synerpair_vec_df['Filename'] == filename]['pairVec'].values[0]
    
    # Convert pairVec string to list
    pair_vec = eval(pair_vec)

    valid_indexes = [idx for idx, val in enumerate(pair_vec) if val > 0]

    # Find the synerpair for each valid index
    synerpairs = [index_to_synerpair[idx] for idx in valid_indexes]
    
    # Initialize an empty set if the Cluster Label is not in the dictionary
    if cluster_label not in cluster_synerpairs:
        cluster_synerpairs[cluster_label] = set()
    
    # Add synerpairs to the set, automatically removing duplicates
    cluster_synerpairs[cluster_label].update(synerpairs)

# Convert the dictionary to a DataFrame
result_df = pd.DataFrame({
    'Cluster Label': cluster_synerpairs.keys(),
    'Synerpairs': [list(synerpairs) for synerpairs in cluster_synerpairs.values()]
})

# Sort by Cluster Label
result_df = result_df.sort_values('Cluster Label')

# Write to a new CSV file
result_df.to_csv(output_path, index=False)

print("Cluster synerpairs have been successfully saved to Phase2_Cluster_Synerpairs.csv")
