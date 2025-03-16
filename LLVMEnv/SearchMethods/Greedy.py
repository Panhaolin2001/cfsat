from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from LLVMEnv.common import get_instrcount

def LeverageSyner_Greedy(SynerList, ll_code, llvm_tools_path):

        all_instr_counts = []
        current_passes = []
        total_get_instrcount_calls = 0

        def evaluate_pass_sequence(pass_sequence):
            nonlocal total_get_instrcount_calls
            try:
                instr_count = get_instrcount(ll_code, pass_sequence, llvm_tools_path=llvm_tools_path)
                total_get_instrcount_calls += 1 
            except Exception as e:
                print(f"Error during instr_count: {e}")
                return float('inf')
            return instr_count

        with ThreadPoolExecutor() as executor:
            instr_counts = list(executor.map(lambda x: evaluate_pass_sequence([x[0], x[1]]), SynerList))
        
        all_instr_counts.extend(instr_counts)
        min_instr_count = min(instr_counts)
        start_pass_index = instr_counts.index(min_instr_count)
        start_pass = list(SynerList[start_pass_index])
        current_passes = start_pass

        while True:
            next_pass = None
            min_value = float('inf')

            with ThreadPoolExecutor() as executor:
                pass_sequences = [current_passes + [pass2] for (pass1, pass2) in SynerList if pass1 == current_passes[-1]]
                instr_counts = list(executor.map(evaluate_pass_sequence, pass_sequences))

            all_instr_counts.extend(instr_counts)
            if instr_counts:
                min_value = min(instr_counts)
                next_pass = pass_sequences[instr_counts.index(min_value)][-1]

            if next_pass and min_value < float('inf'):
                current_passes.append(next_pass)
            else:
                break

            if len(current_passes) == 20:
                break

        min_value = min(all_instr_counts)
        return min_value