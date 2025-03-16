from LLVMEnv.obsUtility.Autophase import get_autophase_obs
from LLVMEnv.actionspace.llvm10_0_0.actions import Actions_LLVM_10_0_0
from LLVMEnv.actionspace.rl_helper import Syner_Action_rl
from LLVMEnv.common import get_instrcount, get_codesize, GenerateOptimizedLLCode

import shlex
import os
import csv
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict

class LLVMEnv(gym.Env):
    def __init__(self, config):
        super(LLVMEnv, self).__init__()
        self._config = config
        self._llvm_tools_path = self._config['llvm_tools_path']
        self._llvm_version = self._config['llvm_version']
        self._reward_type = self._config['reward_space']
        self._reward_baseline = self._config['reward_baseline']
        self._max_steps = self._config['max_steps']
        self.old_pm_llvm_versions = ["llvm-10.0.0", "llvm-10.x"]
        self.csv_path = self._config['csv_path']
        self.Actions = Actions_LLVM_10_0_0
        self.output_dim = self._get_output_dim(self.Actions)
        
        self.benchmark_iterator = None

        if 'dataset' not in self._config:
            self.dataset = None
        else:
            self.dataset = self._config['dataset']

        if 'source_file' not in self._config:
            dataset_path = self.dataset
            files_in_dataset = os.listdir(dataset_path)
            
            if files_in_dataset:
                self._config['source_file'] = os.path.join(dataset_path, files_in_dataset[0])
        
        self.file = os.path.basename(self._config['source_file'])

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)

        self.label_dict = self.csv_to_dict(f"/{current_directory}/../../output/Phase4_graph_test_Sel_Label.csv")
        self.file_synerpairlist_label = self.label_dict[self.file]
        
        self.syner_action = Syner_Action_rl(self.csv_path)
        self.syner_action_pair_ori = self.syner_action.syner_pairlist_dict[self.file_synerpairlist_label]
        self.action_space = Discrete(1000)
        self.syner_action_pair = self.syner_action_pair_ori

        self._ll_code = self.benchmark(self._config['source_file'])
        self._original_ll_code = self._ll_code
        self.feature_dim = self._get_input_dim(self._original_ll_code)
        self.observation_space = self._get_observation_space()
        
        self._optimization_flags = ''
        
    def step(self, action_idx):
        self._steps += 1
        self._applied_passes.append(self._get_action_pair_str(self._steps, action_idx))

        self._optimization_flags = (
            "--enable-new-pm=0 " + " ".join([act for act in self._applied_passes])
            if self._llvm_version not in self.old_pm_llvm_versions
            else " ".join([act for act in self._applied_passes])
        )

        self.update_state(action_idx)

        current_perf = self.calculate_current_perf(self._original_ll_code, self._reward_type, self._optimization_flags)
        self._reward = (self._current_perf - current_perf) / self.baseline_perf
        self._current_perf = current_perf

        action_mask = self._get_action_mask(action_idx)
        terminated = self.is_terminated(self._steps, self._max_steps)
        return {
                "observations": self._state, \
                "action_mask": action_mask,
               }, \
               self._reward, terminated, False, {}

    def is_terminated(self, steps, max_steps):
        return steps >= max_steps or np.all(np.not_equal(self._action_mask, 1))

    def benchmark(self, source_file):
        with open(source_file, 'r') as ll_file:
            ll_code = ll_file.read()
        return ll_code

    def command_line(self):
        return self._optimization_flags
    
    def csv_to_dict(self, file_path):
        result_dict = {}
        with open(file_path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)

            for row in csv_reader:
                value = int(row[1])
                key = row[0].split('/')[-1]
                result_dict[key] = value

        return result_dict
    
    def getActions(self):
        return self.Actions

    def reset(self, *, seed=None, options=None, benchmark=None):
        if benchmark is not None:
            return self._reset(benchmark=benchmark)
        elif self.dataset is not None:
            if self.benchmark_iterator is None:
                self.benchmark_iterator = self.get_benchmarks_from_directory(self.dataset)

            try:
                next_benchmark = next(self.benchmark_iterator)
            except StopIteration:
                print("No more benchmarks in the directory.")
                next_benchmark = None

            return self._reset(benchmark=next_benchmark)
        else:
            return self._reset(benchmark=self._config["source_file"])

    def get_benchmarks_from_directory(self, directory):
        while True:
            try:
                for entry in os.listdir(directory):
                    entry_path = os.path.join(directory, entry)

                    if os.path.isfile(entry_path):
                        yield entry_path
            except OSError as e:
                print(f"Error while listing directory {directory}: {e}")

    def _reset(self, benchmark=None):
        self._reward = 0
        self._steps = 0
        self._ll_code = self.benchmark(benchmark)
        self._original_ll_code = self._ll_code
        self.baseline_perf = self.calculate_baseline_perf(self._original_ll_code, self._reward_baseline)
        self._current_perf = self.baseline_perf
        self._applied_passes = []
        self._datalist = []
        self._optimization_flags = ""
        self.unique_sha1_values = set()
        self.feature_dim = self._get_input_dim(self._original_ll_code)
        self._state = self.init_state()

        self.file = os.path.basename(benchmark)
        self.file_synerpairlist_label = self.label_dict[self.file]
        
        self.syner_action_pair_ori = self.syner_action.syner_pairlist_dict[self.file_synerpairlist_label]
        self.syner_action_pair = self.syner_action_pair_ori

        self._action_mask = np.ones(self.action_space.n)
        action_len = len(self.syner_action_pair)
        for i in range(self.action_space.n - action_len):
            self._action_mask[self.action_space.n - 1 - i] = 0.0


        return {
                "observations": self._state, 
                "action_mask": self._action_mask,
               }, {}
    
    def get_input_dim(self):
        return self.feature_dim
    
    def _get_input_dim(self, ll_code):
        return len(np.array([value for value in self._get_features(ll_code)], dtype=np.float32))
    
    def get_output_dim(self):
        return self.output_dim

    def _get_output_dim(self, Actions):
        return len(list(Actions))

    def _get_observation_space(self):
        return Dict({
            "action_mask":  Box(0.0, 1.0, shape=(self.action_space.n,)),
            "observations": Box(low=float('-inf'), high=float('inf'), shape=(self.feature_dim,), dtype=np.float32)
            })
    
    def _get_next_syner_pair_list(self, pass_name):
        next_pair_list = []
        for action_sub_pair in self.syner_action_pair_ori:
            if pass_name == action_sub_pair[0]:
                next_pair_list.append(action_sub_pair)
        
        return next_pair_list

    def _get_action_mask(self, action_idx):
        self._action_mask = np.ones(self.action_space.n)
        self.syner_action_pair = self._get_next_syner_pair_list(self.syner_action_pair[action_idx][1])
        action_len = len(self.syner_action_pair)
        for i in range(self.action_space.n - action_len):
            self._action_mask[self.action_space.n - 1 - i] = 0.0
        return self._action_mask

    def calculate_baseline_perf(self, ll_code, reward_baseline):
        reward_functions = {
            "IRInstCountOz": lambda: get_instrcount(ll_code, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO3": lambda: get_instrcount(ll_code, "-O3", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO2": lambda: get_instrcount(ll_code, "-O2", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO1": lambda: get_instrcount(ll_code, "-O1", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO0": lambda: get_instrcount(ll_code, "-O0", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeOz": lambda: get_codesize(ll_code, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO3": lambda: get_codesize(ll_code, "-O3", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO2": lambda: get_codesize(ll_code, "-O2", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO1": lambda: get_codesize(ll_code, "-O1", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO0": lambda: get_codesize(ll_code, "-O0", llvm_tools_path=self._llvm_tools_path),
        }

        reward_function = reward_functions.get(reward_baseline)
        if reward_function is None:
            raise ValueError(f"Unknown reward_baseline: {reward_baseline}, please choose 'IRInstCountOz', 'IRInstCountO3', \
                             'IRInstCountO2','IRInstCountO1','IRInstCountO0','CodeSizeOz', 'CodeSizeO3','CodeSizeO2', \
                             'CodeSizeO1','CodeSizeO0','RunTimeOz','RunTimeO3','RunTimeO2','RunTimeO1','RunTimeO0'")

        baseline_perf = reward_function()
        return baseline_perf

    def calculate_current_perf(self, ll_code, reward_type, optimization_flags):
        perf_functions = {
            "IRInstCount": lambda: get_instrcount(ll_code, optimization_flags.split(), llvm_tools_path=self._llvm_tools_path),
            "CodeSize": lambda: get_codesize(ll_code, optimization_flags.split(), llvm_tools_path=self._llvm_tools_path),
        }

        perf_function = perf_functions.get(reward_type)
        if perf_function is None:
            raise ValueError(f"Unknown reward type: {reward_type}, please choose 'IRInstCount', 'CodeSize', 'RunTime'")

        current_perf = perf_function()
        return current_perf

    def update_state(self, action_idx):
        self._ll_code = GenerateOptimizedLLCode(self._original_ll_code, shlex.split(self._optimization_flags), self._llvm_tools_path)
        self._state = np.array([value for value in self._get_features(self._ll_code)], dtype=np.float32)

    def _get_action_pair_str(self, steps, action_idx):
        action_pair_str = ''
        if steps == 1:
            action_pair_str = self.syner_action_pair[action_idx][0] + ' ' + self.syner_action_pair[action_idx][1]
        else:
            action_pair_str = self.syner_action_pair[action_idx][1]
        return action_pair_str

    def _get_features(self, ll_code):
        return get_autophase_obs(ll_code, self._llvm_version)

    def init_state(self):
        state = np.array([value for value in self._get_features(self._ll_code)], dtype=np.float32)
        return state
