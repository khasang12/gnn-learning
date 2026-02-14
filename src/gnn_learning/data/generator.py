import random
import torch
import numpy as np
from torch_geometric.data import Data

def generate_synthetic_rec_data(n_users=100, n_items=50, edge_prob=0.1, train_ratio=0.8, seed=42):
    """
    Generate synthetic user-item interaction data as a bipartite graph.
    Returns PyG Data object with edge_index (train and test edges separated).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Node indices: users 0..n_users-1, items n_users..n_users+n_items-1
    user_nodes = list(range(n_users))
    item_nodes = list(range(n_users, n_users + n_items))

    # Generate all possible edges
    all_pairs = [(u, i) for u in user_nodes for i in item_nodes]
    # Randomly select edges with probability edge_prob
    edges = [pair for pair in all_pairs if random.random() < edge_prob]
    if len(edges) < 10:
        # Ensure minimum edges
        edges = random.sample(all_pairs, min(10, len(all_pairs)))

    # Split edges into train/test
    random.shuffle(edges)
    split_idx = int(len(edges) * train_ratio)
    train_edges = edges[:split_idx]
    test_edges = edges[split_idx:]

    # Convert to edge_index format (2 x num_edges)
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

    # Node features: one-hot identity (optional) or random
    # For simplicity, use random features
    num_nodes = n_users + n_items
    x = torch.randn(num_nodes, 16)  # 16-dimensional random features

    # Create PyG Data object
    data = Data(x=x, edge_index=train_edge_index)
    data.num_users = n_users
    data.num_items = n_items
    data.test_edge_index = test_edge_index
    data.train_edge_index = train_edge_index

    return data
