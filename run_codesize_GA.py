import os
import argparse
import pandas as pd
from LLVMEnv.SearchMethods.GA import LeverageSyner_GA_codesize
from LLVMEnv.actionspace.codesize_pairs import codesize_pairlist

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name, it includes: blas-v0,cbench-v1,mibench-v1,npb-v0,opencv-v0")
args = parser.parse_args()

dataset = args.dataset
llvm_tools_path = "./llvm_tools/"
ll_code_dir = "./dataset/test/" + dataset + "/"

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

df = pd.read_csv(f"/{current_directory}/output/Phase4_graph_test_Sel_Label.csv")
filenames = df['Filename'].tolist()

all = []

for filename in filenames:

    if filename.split("/")[0] == dataset:

        filename = filename.split("/")[1]
        with open(ll_code_dir + filename, 'r') as ll_file:
            ll_code = ll_file.read()

        print("Current File:", filename)  
        filtered_df = df[df['Filename'] == dataset + "/" + filename]
        label = filtered_df['Cluster Label'].tolist()[0]
        score = LeverageSyner_GA_codesize(codesize_pairlist, ll_code, llvm_tools_path)
        all.append(score)
        print("Mean: ", sum(all) / len(all))
        print()

