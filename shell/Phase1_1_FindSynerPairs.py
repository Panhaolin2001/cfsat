import sys
import os
import pandas as pd

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from LLVMEnv.utility.PassSyner import PassSyner

# Get the LLVM tools path from environment variables
llvm_tools_path = "../llvm_tools/"
if llvm_tools_path is None:
    raise EnvironmentError("LLVM_TOOLS_PATH environment variable is not set.")

# Define the directory containing .ll files
train_dataset_dir = os.path.join(project_root, 'dataset', 'train')


# Define the output directory
output_path = os.path.join(project_root, 'output', 'Phase1_SynerPairLists.csv')

syner = PassSyner(train_dataset_dir, llvm_tools_path)
syner.FindSynerPasses(output_path)
