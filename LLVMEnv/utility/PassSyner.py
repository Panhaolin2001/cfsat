import os
import csv
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from LLVMEnv.common import get_instrcount
from LLVMEnv.actionspace.llvm10_0_0.actions import Actions_LLVM_10_0_0

class PassSyner:
    def __init__(self, datasetpath, llvm_tools_path, num_works = 4):
        self.datasetpath = datasetpath
        self.llvm_version = 'llvm-10.0.0'
        self.llvm_tools_path = llvm_tools_path
        self.Passes = self._get_action_values(self.llvm_version)
        self.num_works = num_works
        self.lock = threading.Lock()

    def _process_file(self, filepath, output_csv_path):
        with open(filepath, 'r') as ll_file:
            ll_code = ll_file.read()
        print("Processing:", filepath)

        original_codesize = get_instrcount(ll_code, [], llvm_tools_path=self.llvm_tools_path)
        syner_passpairs = []
        action_aval = []

        sinpass_ic = {}

        # find valid passes
        for action in self.Passes:
            action_ic = get_instrcount(ll_code, [action], llvm_tools_path=self.llvm_tools_path)
            sinpass_ic.update({action: action_ic})
            code_size_change = original_codesize - action_ic
            if code_size_change > 0:
                action_aval.append(action)

        for action2 in action_aval:
            for action1 in self.Passes:
                # if action1 != action2:
                    code_size_change = sinpass_ic.get(action2) - \
                                       get_instrcount(ll_code, [action1, action2], llvm_tools_path=self.llvm_tools_path)
                    if code_size_change > 0:
                        syner_passpairs.append((action1, action2))

        print(f"The number of syner_passpairs in {filepath}: " , len(syner_passpairs))
        
        filename = os.path.basename(filepath)

        with self.lock:
            with open(output_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, syner_passpairs])

    def FindSynerPasses(self, output_csv_path):
        ll_files = []
        for root, dirs, files in os.walk(self.datasetpath):
            for file in files:
                if file.endswith(".ll"):
                    ll_files.append(os.path.join(root, file))

        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Synerpairlist'])

        with ThreadPoolExecutor(max_workers=self.num_works) as executor:
            futures = [executor.submit(self._process_file, filepath, output_csv_path) for filepath in ll_files]
            for future in futures:
                future.result()

        print(f"Results saved to {output_csv_path}")
    
    def _select_actions(self, llvm_version):
        action_space_mappings = {
            "llvm-10.0.0": Actions_LLVM_10_0_0
        }
        selected_actions = action_space_mappings.get(llvm_version)
        if selected_actions is None:
            raise ValueError(f"Unknown action space: {llvm_version}, please choose 'llvm-16.x', 'llvm-14.x', 'llvm-10.0.0' ")
        return selected_actions
    
    def _get_action_values(self, llvm_version):
        actions_enum = self._select_actions(llvm_version)
        return [action.value for action in actions_enum]