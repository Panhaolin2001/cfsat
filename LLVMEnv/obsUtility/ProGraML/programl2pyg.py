import os, sys
import programl as pg

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
sys_root = os.path.dirname(current_file_path)
sys.path.append(sys_root)

from parsing_utils import FeatureExtractor
from pyg_utils import dgl2pyg


def GetProGraMLpyg(ir_code, vocab_path, mode = 0):
    """
    mode : 0, train, update vocab
           1, test, don't update vocab
    """
    G = pg.from_llvm_ir(ir_code)
    graph = pg.to_networkx(G)
    
    if mode == 0:
        extractor = FeatureExtractor(vocab_csv_path=vocab_path, online_update=True)
    else:
        extractor = FeatureExtractor(vocab_csv_path=vocab_path, online_update=False)

    dgl_graph = extractor.process_nx_graph(graph)
    pyg_graph = dgl2pyg(dgl_graph)

    return pyg_graph
