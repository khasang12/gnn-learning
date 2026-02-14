#!/usr/bin/env python3
"""
Quick test to verify the GNN example works functionality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_learning.models.gcn import GCN
from torch_geometric.data import Data
import sys

print("Testing GNN implementation...")

# Test 1: PyTorch imports
print("✓ PyTorch version:", torch.__version__)

# Create dummy graph data
num_nodes = 10
num_features = 5
num_classes = 3

x = torch.randn(num_nodes, num_features)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# Test model from package
try:
    model = GCN(num_features, 16, num_classes)
    out = model(data)

    print(f"✓ Created GCN model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    
    # Test 3: Forward pass with softmax
    # GCN returns log_softmax already
    probs = out
    print(f"✓ Output shape: {probs.shape}")
    
    # Test 4: Training step simulation
    criterion = nn.NLLLoss()
    labels = torch.randint(0, num_classes, (num_nodes,))
    loss = criterion(probs, labels)
    loss.backward()
    
    print(f"✓ Loss computation: {loss.item():.4f}")
    print(f"✓ Gradients exist for conv: {any(p.grad is not None for p in model.gcn1.parameters())}")
    
    print("\nAll basic GNN tests passed!")

except ImportError as e:
    print(f"✗ Failed to import GCN from package: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)