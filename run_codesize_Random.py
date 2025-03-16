import os
import argparse
import pandas as pd
from LLVMEnv.common import get_codesize
from LLVMEnv.SearchMethods.RandomWalk import generate_population
from LLVMEnv.actionspace.codesize_pairs import codesize_pairlist

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name, it includes: blas-v0,cbench-v1,mibench-v1,npb-v0,opencv-v0")
args = parser.parse_args()

dataset = args.dataset
llvm_tools_path = "./llvm_tools/"
ll_code_dir = "./dataset/test/" + dataset + "/"
population = generate_population(codesize_pairlist, 1000)

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

df = pd.read_csv(f"/{current_directory}/output/Phase4_graph_test_Sel_Label.csv")
filenames = df['Filename'].tolist()

codesize_mean = []
filecodesize_mean = []

import concurrent.futures

for filename in filenames:
    if filename.split("/")[0] == dataset:
        filename = filename.split("/")[1]
        with open(ll_code_dir + filename, 'r') as ll_file:
            ll_code = ll_file.read()
        
        Oz = get_codesize(ll_code, ["-Oz"], llvm_tools_path=llvm_tools_path)
        
        print("Current File:", filename)
        filtered_df = df[df['Filename'] == dataset + "/" + filename]
        
        filecodesize_mean = []

        def process_list(opt_list):
            codesize = get_codesize(ll_code, opt_list, llvm_tools_path=llvm_tools_path)
            return (Oz - codesize) / Oz
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_list, population)
        
        filecodesize_mean.extend(results)
        
        filecodesize = max(filecodesize_mean)
        codesize_mean.append(filecodesize)
        filecodesize = []
        print("Mean: ", sum(codesize_mean) / len(codesize_mean))