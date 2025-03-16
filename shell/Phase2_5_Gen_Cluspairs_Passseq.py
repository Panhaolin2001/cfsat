import csv
import ast
import os, sys

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from LLVMEnv.SearchMethods.RandomWalk import generate_population

input_file = os.path.join(project_root, 'output', 'Phase2_Cluster_Synerpairs.csv')
output_file = os.path.join(project_root, 'output', 'Phase2_Cluspairs_Passseq.csv')

with open(input_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['Cluster Label', 'Pass Sequences']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        cluster_label = row['Cluster Label']
        synerpairs = ast.literal_eval(row['Synerpairs'])
        
        population = generate_population(synerpairs, size=100)

        for path in population:
            writer.writerow({'Cluster Label': cluster_label, 'Pass Sequences': path})
