import csv
import os
import sys
import ast
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)
from LLVMEnv.common import get_instrcount
from LLVMEnv.SearchMethods.Greedy import LeverageSyner_Greedy

# Get llvm_tools_path from environment variables
llvm_tools_path = "../llvm_tools/"
if llvm_tools_path is None:
    raise EnvironmentError("LLVM_TOOLS_PATH environment variable is not set.")

def load_labels(file_path):
    """Read CSV file and return a dictionary with Filename as key and Cluster Label as value."""
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        return {rows[0]: int(rows[1]) for rows in reader}

def load_cluster_pass_sequences(file_path):
    """Read pass sequences for each cluster label from CSV and return a dictionary."""
    cluster_passes = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for rows in reader:
            cluster_label = int(rows[0])
            pass_sequences = ast.literal_eval(rows[1])
            if cluster_label not in cluster_passes:
                cluster_passes[cluster_label] = []
            cluster_passes[cluster_label].append(pass_sequences)
    return cluster_passes

def load_edges(file_path):
    # Read Synerpairs CSV file and return a dictionary with Cluster Label as key and Synerpairs list as value
    edges_dict = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for index, rows in enumerate(reader):
            # Convert string representation of list to actual list
            edges_dict[int(rows[0])] = ast.literal_eval(rows[1])
    return edges_dict

def calculate_overoz_for_file(filename, cluster_label, edges_dict, cluster_pass_sequences, base_path, llvm_tools_path):
    """Calculate the best overoz for a single file."""
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        ll_code = file.read()

    pass_sequences = cluster_pass_sequences[cluster_label]
    edges = edges_dict[cluster_label]
    min_instr_count = LeverageSyner_Greedy(edges, ll_code, llvm_tools_path)
    Oz = get_instrcount(ll_code, ['-Oz'], llvm_tools_path=llvm_tools_path)

    overoz = (Oz - min_instr_count) / Oz
    
    # Print filename and corresponding overoz
    print(f"Filename: {filename}, OverOz: {overoz}")

    return filename, overoz

def evaluate_overoz_for_datasets(phase4_labels, edges_dict, cluster_pass_sequences, base_path, llvm_tools_path, max_workers=4):
    """Calculate the best overoz for each file and compute average scores for each dataset."""
    dataset_overozs = {}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(calculate_overoz_for_file, filename, cluster_label, edges_dict, cluster_pass_sequences, base_path, llvm_tools_path)
            for filename, cluster_label in phase4_labels.items()
        ]

        for future in as_completed(futures):
            filename, overoz = future.result()

            dataset_name = filename.split('/')[0]
            if dataset_name not in dataset_overozs:
                dataset_overozs[dataset_name] = []
            dataset_overozs[dataset_name].append(overoz)
    
    meanoveroz = {dataset: mean(overozs) for dataset, overozs in dataset_overozs.items()}
    total_meanoveroz = mean(meanoveroz.values())
    
    return meanoveroz, total_meanoveroz

def save_mean_overoz_to_csv(meanoveroz, total_meanoveroz, output_file):
    """Save the average overoz scores of each dataset and the total average score to a CSV file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'MeanOverOz'])
        for dataset, mean_overoz in meanoveroz.items():
            writer.writerow([dataset, mean_overoz])
        writer.writerow(['Total MeanOverOz', total_meanoveroz])

# File paths
phase4_file_path = os.path.join(project_root, 'output', 'Phase4_graph_test_Sel_Label.csv')
phase2_file_path = os.path.join(project_root, 'output', 'Phase2_Cluspairs_Passseq.csv')
edges_file_path = os.path.join(project_root, 'output', 'Phase2_Cluster_Synerpairs.csv')
base_path = os.path.join(project_root, 'dataset', 'test')
output_file_path = os.path.join(project_root, 'output', 'Phase4_geannvp_GreedyResult.csv')

# Read data from files
phase4_labels = load_labels(phase4_file_path)
edges_dict = load_edges(edges_file_path)
cluster_pass_sequences = load_cluster_pass_sequences(phase2_file_path)

# Calculate meanoveroz
meanoveroz, total_meanoveroz = evaluate_overoz_for_datasets(phase4_labels, edges_dict, cluster_pass_sequences, base_path, llvm_tools_path)

# Save scores to file
save_mean_overoz_to_csv(meanoveroz, total_meanoveroz, output_file_path)

print("Scores have been calculated and saved successfully.")
