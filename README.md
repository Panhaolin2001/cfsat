# 🚀 CFSAT: Compiler Flags Synergistic Auto-Tuning

The following experimental scripts contain **only the experimental procedures and results of the methods** proposed in the paper, and **do not contain the experiments and results of the other techniques compared**.

## Directory Tree
```
├── DLL                              # Dynamic link libraries for program representation
│   ├── libAutophase_10_0_0.so
│   └── libInstCount_10_0_0.so
├── Getting_Start_Guide.md
├── LLVMEnv                          # Tuning related internal code
│   ├── SearchMethods
│   ├── actionspace
│   ├── common.py
│   ├── env
│   ├── net
│   ├── obsUtility
│   └── utility
├── dataset                          # Datasets, including training set validation set and test set
│   ├── test
│   ├── train
│   └── val
├── directory_tree.txt
├── llvm_tools                       # LLVM tools used for tuning
│   ├── clang++
│   ├── llvm-size
│   └── opt
├── output                           # All outputs from the tuning process for the number of instructions
│   ├── Phase1_Filtered_SynerPairLists.csv
│   ├── Phase2_Cluspairs_Passseq.csv
│   ├── Phase2_Cluster_Synerpairs.csv
│   ├── Phase2_Enumerated_pairs.csv
│   ├── Phase2_Synerpair_Vec.csv
│   ├── Phase2_file_Kmeans100.csv
│   ├── Phase3_graph_trained_model.pth
│   ├── Phase3_test_pyg_dataset.pt
│   ├── Phase3_train_pyg_dataset.pt
│   ├── Phase3_train_random_Set_Vectors.csv
│   ├── Phase3_val_pyg_dataset.pt
│   ├── Phase3_val_random_Set_Vectors.csv
│   ├── Phase4_geannvp_100Result.csv       # Corresponds to the Random column of Table 4 in the paper.
│   ├── Phase4_geannvp_GAResult.csv        # Corresponds to the GA column of Table 4 in the paper.
│   ├── Phase4_geannvp_GreedyResult.csv    # Corresponds to the Greedy column of Table 4 in the paper.
│   ├── Phase4_graph_test_Sel_Label.csv
│   └── vocab.csv
├── run_codesize_GA.py              # Running script for GA(Our) column for Table5 in the paper.
├── run_codesize_Random.py          # Running script for Random(Our) column for Table5 in the paper.
├── run_rl.py                       # Running script for RL column for Table4 in the paper.
└── shell                           # Scripts of all reproduction steps of the methods described in the paper.
    ├── Phase1_1_FindSynerPairs.py
    ├── Phase1_2_FilterSynerPair.py
    ├── Phase2_1_Enum_Synerpairs.py
    ├── Phase2_2_Gen_file_Pairvec.py
    ├── Phase2_3_Kmeans100.py
    ├── Phase2_4_Gen_Cluster_Synpairs.py
    ├── Phase2_5_Gen_Cluspairs_Passseq.py
    ├── Phase3_1_file_random_setVec.py
    ├── Phase3_2_GenPygDateset.py
    ├── Phase3_3_graph_train.py
    ├── Phase4_1_graph_SelectLabel.py
    ├── Phase4_2_geannvp_100test.py     # Running script for Random column for Table4 in the paper.
    ├── Phase4_2_geannvp_GAtest.py      # Running script for GA column for Table4 in the paper.
    └── Phase4_2_geannvp_Greedytest.py  # Running script for Greedy column for Table4 in the paper.
```

## How to reproduce main results in the paper?
The main experimental part of the paper is in Table4 and Table5, and the following mainly describes the specific acquisition method of experimental data in Table4 and Table5.
First, enter the compilerautotuning conda environment:
```conda activate cgo_paper323```.
### About Table4
1. Random method: 
    ``` 
    cd ./shell/
    python Phase4_2_geannvp_100test.py
    ```
    The results are saved to *./output/Phase4_geannvp_100Result.csv*
2. Greedy method:
    ``` 
    cd ./shell/
    python Phase4_2_geannvp_Greedytest.py
    ```
    The results are saved to *./output/Phase4_geannvp_GreedyResult.csv*
3. GA method:
    ``` 
    cd ./shell/
    python Phase4_2_geannvp_GAtest.py
    ```
    The results are saved to *./output/Phase4_geannvp_GAResult.csv*
3. RL method:
    ``` 
    python run_rl.py --filepath XXX
    ```
    XXX is the specific file path. For example, `python run_rl.py --filepath cbench-v1/cbench-v1_adpcm.ll`. The result of a single file run are obtained via *episode_reward_max* output by rllib.

### About Table5

1. Random method: 
    ``` 
    python run_codesize_Random.py --dataset XXX
    ```

2. GA method:
    ``` 
    python run_codesize_GA.py --dataset XXX
    ```
    
*XXX* can be **blas-v0, cbench-v1, mibench-v1, npb-v0, opencv-v0**. For example, `python run_codesize_Random.py --dataset cbench-v1` or `python run_codesize_GA.py --dataset cbench-v1`. The output is shown below, where *Current File* is the benchmark that is currently being tuned, and *Mean* is the total average codesize reduction percentage (compared to Oz) for all benchmarks that have been tuned up to now. **The last Mean output is the average percentage reduction in codesize for all benchmarks in this dataset compared to Oz.**

```
Current File: cbench-v1_adpcm.ll
Mean:  0.2694886839899413
Current File: cbench-v1_bitcount.ll
Mean:  0.20702907780710372
Current File: cbench-v1_blowfish.ll
Mean:  0.30150201774586927
Current File: cbench-v1_crc32.ll
Mean:  0.3876450769125669
Current File: cbench-v1_dijkstra.ll
Mean:  0.32051575109155916
...
```


*NOTE: If you only want to reproduce the main results, please follow the instructions above to run the script. If you want to reproduce the entire technical process, please run the script according to PhaseX_x inside the shell directory.*

## Citing this work
If you use this work, please cite:
```bibtex
@inproceedings{pan2025towards,
  title={Towards Efficient Compiler Auto-tuning: Leveraging Synergistic Search Spaces},
  author={Pan, Haolin and Wei, Yuanyu and Xing, Mingjie and Wu, Yanjun and Zhao, Chen},
  booktitle={Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization},
  pages={614--627},
  year={2025}
}
