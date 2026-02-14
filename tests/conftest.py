"""
Pytest fixtures for GNN Learning tests.
"""
import pytest
import torch
import numpy as np
from torch_geometric.data import Data


@pytest.fixture
def dummy_graph_data():
    """Create a small dummy graph for testing GCN models."""
    num_nodes = 10
    num_features = 5
    num_classes = 3

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]], dtype=torch.long)
    y = torch.randint(0, num_classes, (num_nodes,))

    # Add train/val/test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:7] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[7:9] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[9] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data


@pytest.fixture
def synthetic_rec_data():
    """Create synthetic recommendation data for testing LightGCN."""
    from gnn_learning.data.generator import generate_synthetic_rec_data
    data = generate_synthetic_rec_data(n_users=20, n_items=15, seed=42)
    return data


@pytest.fixture
def device():
    """Get device for testing (CPU only to avoid CUDA dependencies)."""
    return torch.device('cpu')