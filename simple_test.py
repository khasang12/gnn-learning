#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

print("Testing environment...")
print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try importing torch_geometric
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    print(f"PyTorch Geometric available")
    
    # Simple test
    model = GCNConv(5, 3)
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,0],[1,2,3,4,5,6,7,8,9,9]], dtype=torch.long)
    out = model(x, edge_index)
    print(f"GCNConv test passed: input {x.shape}, output {out.shape}")
    
except Exception as e:
    print(f"PyTorch Geometric import failed: {e}")
    sys.exit(1)

print("All imports successful!")