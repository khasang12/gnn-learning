#!/usr/bin/env python3
"""
Quick test to verify the GNN example works without downloading full dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys

print("Testing GNN implementation...")

# Test 1: PyTorch imports
print("✓ PyTorch version:", torch.__version__)

# Test 2: GCNConv layer creation
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# Create dummy graph data
num_nodes = 10
num_features = 5
num_classes = 3

x = torch.randn(num_nodes, num_features)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]], dtype=torch.long)

# Test model
model = SimpleGCN(num_features, num_classes)
out = model(x, edge_index)

print(f"✓ Created GCN model with {sum(p.numel() for p in model.parameters())} parameters")
print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {out.shape}")
print(f"✓ Edge index shape: {edge_index.shape}")

# Test 3: Forward pass with softmax
probs = F.log_softmax(out, dim=1)
print(f"✓ Softmax output shape: {probs.shape}")
print(f"✓ Softmax sums per node: {torch.exp(probs).sum(dim=1)}")

# Test 4: Training step simulation
criterion = nn.NLLLoss()
labels = torch.randint(0, num_classes, (num_nodes,))
loss = criterion(probs, labels)
loss.backward()

print(f"✓ Loss computation: {loss.item():.4f}")
print(f"✓ Gradients exist for conv.weight: {model.conv.weight.grad is not None}")

print("\nAll basic GNN tests passed!")
print("\nTo run the full example:")
print("  python gnn_example.py")
print("\nNote: The first run will download the Cora dataset (~1.5MB)")