import os
import sys
import csv
import numpy as np
import torch

from torch_geometric.data import Data, Batch

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
# Add the project root to the Python path
sys.path.append(project_root)

from LLVMEnv.obsUtility.ProGraML.dataset import IndexCustomDataset
from LLVMEnv.net.gean_pyg import GNNEncoder

# Check for available GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure a GPU is available and CUDA is properly installed.")
device = torch.device('cuda')

def process_files(directory, output_csv):
    test_dataset_path = os.path.join(project_root, 'output', 'Phase3_test_pyg_dataset.pt')
    test_dataset = IndexCustomDataset.load(test_dataset_path)

    # Load model parameters
    model = GNNEncoder(node_vocab_size=95).to(device)
    model_file = os.path.join(project_root, 'output', 'Phase3_graph_trained_model.pth')
    model.load_state_dict(torch.load(model_file))
    model.eval()
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Cluster Label'])  # Write CSV header

        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    pyg = IndexCustomDataset.get_by_index(test_dataset, filename)
                    data_list = [pyg]
                    data_batch = Batch.from_data_list(data_list)
                    pyg = data_batch.to(device)
                    
                    with torch.no_grad():
                        output = model(pyg)
                        prediction = torch.exp(output)
                        max_index = int(torch.argmax(prediction))
                        
                    # Get the relative path
                    relative_path = os.path.relpath(file_path, directory)
                    writer.writerow([relative_path, max_index])

# Set the directories to process and the output CSV file paths
test_directory_to_process = os.path.join(project_root, 'dataset', 'test')
test_output_csv_file = os.path.join(project_root, 'output', 'Phase4_graph_test_Sel_Label.csv')

# Process test files and write results to CSV
process_files(test_directory_to_process, test_output_csv_file)
