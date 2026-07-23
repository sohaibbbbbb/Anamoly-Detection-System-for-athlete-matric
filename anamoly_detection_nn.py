import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

class AthleteAnomalyDataset(Dataset):
    """Casts preprocessed numerical matrices into optimized PyTorch Tensors."""
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) 

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class AnomalyDetectorANN(nn.Module):
    """Binary classification network to flag suspicious athlete performance metrics."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return self.sigmoid(out)

def train_anomaly_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    epochs: int, 
    lr: float
) -> nn.Module:
    """Executes the manual backpropagation loop to update network weights."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()           
            predictions = model(batch_X)    
            loss = criterion(predictions, batch_y) 
            loss.backward()                 
            optimizer.step()                
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    return model