import os
import sys

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from LLVMEnv.obsUtility.ProGraML.dataset import CustomDataset, IndexCustomDataset

train_dataset_root = os.path.join(project_root, 'dataset', 'train')
val_dataset_root = os.path.join(project_root, 'dataset', 'val')
test_dataset_root = os.path.join(project_root, 'dataset', 'test')

vocab_path = os.path.join(project_root, 'output', 'vocab.csv')

train_SetVec_path = os.path.join(project_root, 'output', 'Phase3_train_random_Set_Vectors.csv')
val_SetVec_path = os.path.join(project_root, 'output', 'Phase3_val_random_Set_Vectors.csv')

train_pt_path = os.path.join(project_root, 'output', 'Phase3_train_pyg_dataset.pt')
val_pt_path = os.path.join(project_root, 'output', 'Phase3_val_pyg_dataset.pt')
test_pt_path = os.path.join(project_root, 'output', 'Phase3_test_pyg_dataset.pt')

# mode = 0, need train_SetVec_path
train_dataset = CustomDataset(root_dir=train_dataset_root, vocab_path=vocab_path, csv_path=train_SetVec_path, mode=0)
train_dataset.save(train_pt_path)

# mode = 1, need val_SetVec_path
val_dataset = CustomDataset(root_dir=val_dataset_root, vocab_path=vocab_path, csv_path=val_SetVec_path, mode=1)
val_dataset.save(val_pt_path)

# mode = 1, don't need SetVec_path
test_dataset = IndexCustomDataset(root_dir=test_dataset_root, vocab_path=vocab_path)
test_dataset.save(test_pt_path)