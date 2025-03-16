import torch
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from LLVMEnv.net.gean_pyg import GNNEncoder
from LLVMEnv.obsUtility.ProGraML.dataset import CustomDataset

train_dataset_path = os.path.join(project_root, 'output', 'Phase3_train_pyg_dataset.pt')
val_dataset_path = os.path.join(project_root, 'output', 'Phase3_val_pyg_dataset.pt')

train_dataset = CustomDataset.load(train_dataset_path)
val_dataset = CustomDataset.load(val_dataset_path)

output_model_path = os.path.join(project_root, 'output', 'Phase3_graph_trained_model.pth')

def custom_collate_fn(data_list):
    # Use torch_geometric's Batch to handle batch data
    batch = Batch.from_data_list(data_list)
    
    if len(batch.y.shape) == 1:
        # If y has been flattened, reshape it
        batch.y = batch.y.view(-1, 100)  # Assuming each y should have a dimension of 100
    
    return batch


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)

# Check for available GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure a GPU is available and CUDA is properly installed.")
device = torch.device('cuda:1')

model = GNNEncoder(node_vocab_size=95).to(device)

# Define loss function and optimizer
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

# Train the model
num_epochs = 100
T = 0.002

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        y = data.y.to(device)
        y_scaled = y / T
        y_softmax = nn.functional.softmax(y_scaled, dim=-1)
        loss = criterion(output, y_softmax)  
        loss.backward() 
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            y = data.y.to(device)
            y_scaled = y / T
            y_softmax = nn.functional.softmax(y_scaled, dim=-1)
            loss = criterion(output, y_softmax)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    # Update learning rate scheduler
    scheduler.step(val_loss)

# Save the model
torch.save(model.state_dict(), output_model_path)

print("Model has been successfully saved.")