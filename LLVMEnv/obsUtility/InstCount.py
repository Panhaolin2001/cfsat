import ctypes
import os

class InstCountDataStruct(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 64), ("value", ctypes.c_int)]

def get_inst_count_obs(ir_file_path, llvm_version="llvm-10.0.0"):
    project_directory = os.path.dirname(os.path.abspath(__file__))
    library_path = None
    library_path = os.path.join(project_directory, '../../DLL/', 'libInstCount_10_0_0.so')
    
    result_array = (InstCountDataStruct * 70)()
    my_cpp_lib = ctypes.CDLL(library_path)
    my_cpp_lib.GetInstCount(ir_file_path.encode(), result_array)

    result_dict = {item.name.decode(): item.value for item in result_array}

    ic_value = result_dict.get('TotalInsts', None)

    return ic_value
