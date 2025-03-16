import os, sys
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, DataLoader

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
sys_root = os.path.dirname(current_file_path)
sys.path.append(sys_root)

from programl2pyg import GetProGraMLpyg

class CustomDataset(Dataset):
    def __init__(self, root_dir, vocab_path, csv_path = None, mode=0):
        """
        Initialize the custom dataset class
        
        Parameters:
        - root_dir: Directory containing LLVM IR files
        - vocab_path: Path to the vocabulary file
        - csv_path: Path to the CSV file containing feature vectors
        - mode: Mode (0 for training, 1 for testing)
        """
        self.root_dir = root_dir
        self.vocab_path = vocab_path
        self.csv_path = csv_path
        self.mode = mode
        self.files = []
        self.feature_data = None
        if csv_path is not None:
            self.feature_data = pd.read_csv(csv_path)
        
        # Recursively find all LLVM IR files in root_dir
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.ll'):  # Assuming LLVM IR files have a '.ll' extension
                    self.files.append(os.path.join(dirpath, filename))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Get the LLVM IR file path
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)
        
        # Read the content of the LLVM IR file
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Convert LLVM IR content to PyG graph
        pyg_graph = GetProGraMLpyg(file_content, self.vocab_path, mode=self.mode)
        
        if self.feature_data is not None:
            # Extract feature vector from the CSV file
            feature_vector = self.feature_data[self.feature_data['Filename'] == file_name]['Set Vector'].values[0]
            feature_vector = torch.tensor(eval(feature_vector), dtype=torch.float)
            
            # Create Data object
            data = Data(
                x=pyg_graph.x,  # Node feature matrix
                edge_index=pyg_graph.edge_index,
                y=feature_vector,  # Label
                type=pyg_graph.type, 
                function=pyg_graph.function,
                block=pyg_graph.block,
                flow=pyg_graph.flow,
                position=pyg_graph.position
            )
        else:
            data = Data(
                x=pyg_graph.x,  # Node feature matrix
                edge_index=pyg_graph.edge_index,
                type=pyg_graph.type, 
                function=pyg_graph.function,
                block=pyg_graph.block,
                flow=pyg_graph.flow,
                position=pyg_graph.position
            )
        
        return data
    
    def save(self, save_path):
        """
        Save the entire dataset to a single file.
        
        Parameters:
        - save_path: Path where the dataset file will be saved
        """
        # Create the dataset list
        dataset = [self[i] for i in range(len(self))]
        
        # Save the dataset to the specified file
        torch.save(dataset, save_path)

    @staticmethod
    def load(save_path):
        """
        Load the entire dataset from a single file.
        
        Parameters:
        - save_path: Path where the dataset is saved
        
        Returns:
        - dataset: List of PyG Data objects
        """
        dataset = torch.load(save_path)
        return dataset


class IndexCustomDataset(Dataset):
    """
    For test dataset
    """
    def __init__(self, root_dir, vocab_path):
        self.root_dir = root_dir
        self.vocab_path = vocab_path
        self.files = []
        
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.ll'):
                    self.files.append(os.path.join(dirpath, filename))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)
        
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        pyg_graph = GetProGraMLpyg(file_content, self.vocab_path, mode=1)
        
        data = Data(
            x=pyg_graph.x,
            edge_index=pyg_graph.edge_index,
            type=pyg_graph.type, 
            function=pyg_graph.function,
            block=pyg_graph.block,
            flow=pyg_graph.flow,
            position=pyg_graph.position
        )
        
        return data
    
    def save(self, save_path):
        dataset = {}
        for i in range(len(self)):
            file_name = os.path.basename(self.files[i])
            dataset[file_name] = self[i]
        
        torch.save(dataset, save_path)

    @staticmethod
    def load(save_path):
        dataset = torch.load(save_path)
        return dataset
    
    @staticmethod
    def get_by_index(dataset, index):
        """
        Retrieve a Data object from the dataset using the given index.
        
        Parameters:
        - dataset: The loaded dataset (a dictionary)
        - index: The index to look up (e.g., filename)
        
        Returns:
        - data: The corresponding PyG Data object
        """
        return dataset.get(index, None)