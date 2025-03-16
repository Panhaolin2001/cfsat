import os
import sys
import csv
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from LLVMEnv.common import get_instrcount

# Get llvm_tools_path from environment variables
llvm_tools_path = "../llvm_tools/"
if llvm_tools_path is None:
    raise EnvironmentError("LLVM_TOOLS_PATH environment variable is not set.")

def process_file(ll_file_path, pass_sequences_by_label, llvm_tools_path):
    with open(ll_file_path, 'r', encoding='utf-8') as file:
        ll_code = file.read()

    ori = get_instrcount(ll_code, [], llvm_tools_path=llvm_tools_path)
    
    best_overori_by_label = {}
    
    for label, pass_sequences in pass_sequences_by_label.items():
        best_overori = float('-inf')

        for pass_sequence in pass_sequences:
            instr_count = get_instrcount(ll_code, pass_sequence, llvm_tools_path=llvm_tools_path)
            overori = (ori - instr_count) / ori

            if overori > best_overori:
                best_overori = overori

        best_overori_by_label[label] = best_overori

    return best_overori_by_label

def find_ll_files(directory):
    ll_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ll'):
                ll_files.append(os.path.join(root, file))
    return ll_files

def apply_pass_sequences_to_files(directory, csv_file, output_file):
    pass_sequences_by_label = {}

    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = int(row['Cluster Label'])
            pass_sequence = ast.literal_eval(row['Pass Sequences'])
            if label not in pass_sequences_by_label:
                pass_sequences_by_label[label] = []
            pass_sequences_by_label[label].append(pass_sequence)

    ll_files = find_ll_files(directory)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, ll_file, pass_sequences_by_label, llvm_tools_path): ll_file
            for ll_file in ll_files
        }

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Filename', 'Set Vector']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for future in as_completed(futures):
                ll_file = futures[future]
                try:
                    best_overori_by_label = future.result()
                    set_vector = [best_overori_by_label[label] for label in sorted(best_overori_by_label.keys())]
                    filename = os.path.basename(ll_file)
                    writer.writerow({'Filename': filename, 'Set Vector': set_vector})
                except Exception as e:
                    print(f"Error processing {ll_file}: {e}")

    print(f"Results have been saved to {output_file}")

train_dataset = os.path.join(project_root, 'dataset', 'train')
val_dataset = os.path.join(project_root, 'dataset', 'val')
csv_file = os.path.join(project_root, 'output', 'Phase2_Cluspairs_Passseq.csv')
train_output_file = os.path.join(project_root, 'output', 'Phase3_train_random_Set_Vectors.csv')
val_output_file = os.path.join(project_root, 'output', 'Phase3_val_random_Set_Vectors.csv')

apply_pass_sequences_to_files(train_dataset, csv_file, train_output_file)
apply_pass_sequences_to_files(val_dataset, csv_file, val_output_file)
