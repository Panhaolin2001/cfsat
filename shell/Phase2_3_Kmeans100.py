import os
import csv
import ast
import numpy as np
from sklearn.cluster import KMeans

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Define input and output paths
input_path = os.path.join(project_root, 'output', 'Phase2_Synerpair_Vec.csv')
output_path = os.path.join(project_root, 'output', 'Phase2_file_Kmeans100.csv')

# Step 1: Read Phase2_Synerpair_Vec.csv and extract vector data
program_names = []
vectors = []

with open(input_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        program_names.append(row['Filename'])
        vector_str = row['pairVec']
        vector = ast.literal_eval(vector_str)
        vectors.append(vector)

# Convert vector data to numpy array
vectors = np.array(vectors)

# Step 2: Perform clustering using KMeans
n_clusters = 100  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(vectors)
labels = kmeans.predict(vectors)

# Step 3: Save the clustering results to a new CSV file
with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Filename', 'Cluster Label'])
    
    for program_name, label in zip(program_names, labels):
        writer.writerow([program_name, label])

print("Phase2_file_Kmeans100.csv has been successfully generated.")
