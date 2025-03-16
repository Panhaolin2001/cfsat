import os,tempfile
import io
import subprocess
from .obsUtility.InstCount import get_inst_count_obs

def get_codesize(ll_code, *opt_flags, llvm_tools_path=None):
    if llvm_tools_path is None:
        raise ValueError("llvm_tools_path must be provided")

    try:
        # Create temporary files for LLVM IR code, optimized IR code, and object file
        with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as ll_file, \
             tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as opt_file, \
             tempfile.NamedTemporaryFile(suffix=".o", delete=False) as obj_file:
            
            # Write original LLVM IR code to temporary file
            ll_file.write(ll_code.encode())
            ll_file.flush()
            
            # Construct opt command
            opt_path = os.path.join(llvm_tools_path, "opt")
            flat_opt_options = [str(item) for sublist in opt_flags for item in (sublist if isinstance(sublist, list) else [sublist])]
            cmd_opt = [opt_path] + flat_opt_options + ["-S", ll_file.name, "-o", opt_file.name]

            # Run opt to optimize LLVM IR
            subprocess.run(cmd_opt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Read optimized LLVM IR code from temporary file
            with open(opt_file.name, "r") as f:
                optimized_ll_code = f.read()

            # Write optimized LLVM IR code to the temporary file (if not already done)
            with open(opt_file.name, "w") as f:
                f.write(optimized_ll_code)
                f.flush()

            # Construct clang command to compile LLVM IR to object file
            clang_path = os.path.join(llvm_tools_path, "clang++")
            obj_file_path = obj_file.name
            cmd_clang = [clang_path, "-o", obj_file_path, "-c", "-Wno-override-module", opt_file.name]

            # Run clang to compile LLVM IR to object file
            subprocess.run(cmd_clang, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Construct llvm-size command to get the size of the object file
            llvm_size_path = os.path.join(llvm_tools_path, "llvm-size")
            cmd_size = [llvm_size_path, obj_file_path]

            # Run llvm-size and capture the output
            result = subprocess.run(cmd_size, stdout=subprocess.PIPE, check=True)
            output = result.stdout.decode()

        # Clean up temporary files
        os.remove(ll_file.name)
        os.remove(opt_file.name)
        os.remove(obj_file_path)

        # Extract the text size from llvm-size output
        for line in output.splitlines():
            if line.strip().endswith(os.path.basename(obj_file_path)):
                text_size = int(line.split()[0])
                return text_size

        # If we reach this point, something went wrong
        raise RuntimeError("Failed to extract text size from llvm-size output")
    
    except subprocess.CalledProcessError:
        # If any of the subprocess commands fail, return infinity
        return float('inf')
    except Exception:
        # Clean up temporary files if an unexpected error occurs
        try:
            if os.path.exists(ll_file.name):
                os.remove(ll_file.name)
            if os.path.exists(opt_file.name):
                os.remove(opt_file.name)
            if os.path.exists(obj_file_path):
                os.remove(obj_file_path)
        except:
            pass
        return float('inf')

def GenerateOptimizedLLCode(input_code, optimization_options, llvm_tools_path=None):
    try:
        opt_path = os.path.join(llvm_tools_path, "opt") if llvm_tools_path else "opt"
        
        # Flatten the optimization options list
        flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Use io.StringIO to simulate a file-like object
        input_code_io = io.StringIO()
        input_code_io.write(input_code)
        input_code_io.seek(0)  # Reset the file position to the beginning

        # Prepare the command for subprocess
        cmd_opt = [opt_path] + flat_opt_options + ["-S"]

        # Run the opt command with the given input code
        result = subprocess.run(cmd_opt, input=input_code_io.getvalue(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Return the optimized LLVM code
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If there's an error, output the original input code
        # print(f"Error occurred during optimization: {e}")
        # print(f"Standard output: {e.stdout}")
        # print(f"Standard error: {e.stderr}")
        return input_code
    
def get_instrcount(ll_code, *opt_flags, llvm_tools_path=None):

    if llvm_tools_path is None:
        raise ValueError("llvm_tools_path must be provided")
    
    after_ll_code = GenerateOptimizedLLCode(ll_code, opt_flags, llvm_tools_path)

    return get_inst_count_obs(after_ll_code, "llvm-10.0.0")
