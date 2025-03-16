import csv
import os
import sys
import ast
import fnmatch
import pandas as pd

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from LLVMEnv.common import get_instrcount

# Get llvm_tools_path from environment variables
llvm_tools_path = "../llvm_tools/"
if llvm_tools_path is None:
    raise EnvironmentError("LLVM_TOOLS_PATH environment variable is not set.")

# Define paths
Train_dir = os.path.join(project_root, 'dataset', 'train')
Enumerated_pairs_pth = os.path.join(project_root, 'output', 'Phase2_Enumerated_pairs.csv')
Filtered_SynerPairLists_pth = os.path.join(project_root, 'output', 'Phase1_Filtered_SynerPairLists.csv')
output = os.path.join(project_root, 'output', 'Phase2_Synerpair_Vec.csv')

def find_program_file(program_name, train_dir=Train_dir):
    # Traverse the root directory and its subdirectories
    for root, dirs, files in os.walk(train_dir):
        for filename in fnmatch.filter(files, program_name):
            return os.path.join(root, filename)
    return None

# Step 1: Read Phase2_Enumerated_pairs.csv and create a value-to-index mapping
synerpair_to_index = {}
with open(Enumerated_pairs_pth, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        synerpair_to_index[row['synerpair']] = int(row['index'])

# Get the number of rows
df = pd.read_csv(Enumerated_pairs_pth)
row_count = len(df)

# Step 2: Read Phase1_Filtered_SynerPairLists.csv and generate Phase2_Synerpair_Vec.csv
with open(Filtered_SynerPairLists_pth, mode='r', encoding='utf-8') as infile, open(output, mode='w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # Write the header
    writer.writerow(['Filename', 'pairVec'])

    # Process each row of data
    for row in reader:
        program_name = row['Filename']
        synerpair_str = row['Synerpairlist']
        synerpair_list = ast.literal_eval(synerpair_str)

        program_file_path = find_program_file(program_name)
        if program_file_path:
            print(f"Found {program_name} at {program_file_path}")
            # Read the found file
            with open(program_file_path, 'r') as program_file:
                ll_code = program_file.read()
        else:
            print(f"{program_name} not found!")
            continue

        ori_count = get_instrcount(ll_code, [], llvm_tools_path=llvm_tools_path)

        # Create a vector of length row_count initialized to 0
        vector = [0] * row_count
        
        # Set the score at the index of the matched value
        for pair in synerpair_list:
            if str(pair) in synerpair_to_index:
                compile_count = get_instrcount(ll_code, list(pair), llvm_tools_path=llvm_tools_path)
                overori = (ori_count - compile_count) / ori_count
                # Check for values less than or equal to 0
                if overori <= 0:
                    raise ValueError(f"Error: overori is less than or equal to 0 for pair {pair}")
                
                vector[synerpair_to_index[str(pair)]] = overori

        # Write the result
        writer.writerow([program_name, vector])

print("Phase2_Synerpair_Vec.csv has been successfully generated.")
