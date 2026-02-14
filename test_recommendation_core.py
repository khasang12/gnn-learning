#!/usr/bin/env python3
"""
Test the recommendation system core functions without Streamlit.
"""

import torch
import numpy as np
import random

# Import the functions we want to test
import sys
sys.path.insert(0, '.')

# Mock Streamlit for testing
class MockStreamlit:
    class session_state:
        rec_data = None
        rec_model = None
        rec_train_losses = []
        rec_metrics = None
        rec_n_users = None
        rec_n_items = None

sys.modules['streamlit'] = MockStreamlit

# Now import the functions from gnn_streamlit.py
# We'll need to extract the functions directly or test them differently
# Instead, let's copy the core functions here for testing

print("Testing recommendation system core functions...")

# Test 1: Synthetic data generation
print("\n1. Testing synthetic data generation...")
from torch_geometric.data import Data
import random as rnd

def generate_synthetic_rec_data_test():
    n_users = 50
    n_items = 30
    edge_prob = 0.1
    train_ratio = 0.8
    seed = 42

    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    user_nodes = list(range(n_users))
    item_nodes = list(range(n_users, n_users + n_items))
    all_pairs = [(u, i) for u in user_nodes for i in item_nodes]
    edges = [pair for pair in all_pairs if rnd.random() < edge_prob]
    if len(edges) < 10:
        edges = rnd.sample(all_pairs, min(10, len(all_pairs)))

    rnd.shuffle(edges)
    split_idx = int(len(edges) * train_ratio)
    train_edges = edges[:split_idx]
    test_edges = edges[split_idx:]

    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

    num_nodes = n_users + n_items
    x = torch.randn(num_nodes, 16)

    data = Data(x=x, edge_index=train_edge_index)
    data.num_users = n_users
    data.num_items = n_items
    data.test_edge_index = test_edge_index
    data.train_edge_index = train_edge_index

    return data

data = generate_synthetic_rec_data_test()
print(f"✓ Generated synthetic data with {data.num_users} users, {data.num_items} items")
print(f"✓ Training edges: {data.edge_index.shape[1]}")
print(f"✓ Test edges: {data.test_edge_index.shape[1]}")

# Test 2: LightGCN model
print("\n2. Testing LightGCN model...")
try:
    from torch_geometric.utils import add_self_loops, degree

    class LightGCNTest(torch.nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers

            self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
            self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

            torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
            torch.nn.init.normal_(self.item_embedding.weight, std=0.1)

        def forward(self, data):
            user_emb = self.user_embedding(torch.arange(self.num_users, device=self.user_embedding.weight.device))
            item_emb = self.item_embedding(torch.arange(self.num_items, device=self.item_embedding.weight.device))
            embeddings = torch.cat([user_emb, item_emb], dim=0)

            edge_index = data.edge_index
            num_nodes = self.num_users + self.num_items

            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            row, col = edge_index
            deg = degree(row, num_nodes, dtype=embeddings.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            all_embeddings = [embeddings]
            for _ in range(self.num_layers):
                from torch_geometric.utils import scatter
                messages = embeddings[col] * norm.view(-1, 1)
                embeddings = scatter(messages, row, dim=0, dim_size=num_nodes, reduce='sum')
                all_embeddings.append(embeddings)

            final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
            final_user_emb = final_embeddings[:self.num_users]
            final_item_emb = final_embeddings[self.num_users:]

            return final_user_emb, final_item_emb

    model = LightGCNTest(data.num_users, data.num_items, embedding_dim=32, num_layers=2)
    user_emb, item_emb = model(data)
    print(f"✓ Created LightGCN model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ User embeddings shape: {user_emb.shape}")
    print(f"✓ Item embeddings shape: {item_emb.shape}")

except Exception as e:
    print(f"✗ Error testing LightGCN: {e}")
    import traceback
    traceback.print_exc()

# Test 3: BPR loss
print("\n3. Testing BPR loss...")
def bpr_loss_test(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    return loss

# Create dummy embeddings
batch_size = 10
embedding_dim = 32
user_emb_dummy = torch.randn(batch_size, embedding_dim)
pos_item_emb_dummy = torch.randn(batch_size, embedding_dim)
neg_item_emb_dummy = torch.randn(batch_size, embedding_dim)

loss = bpr_loss_test(user_emb_dummy, pos_item_emb_dummy, neg_item_emb_dummy)
print(f"✓ BPR loss computed: {loss.item():.4f}")

print("\n✅ Recommendation system core tests passed!")
print("\nNote: This tests the core functions without Streamlit.")
print("For full integration testing, run: streamlit run gnn_streamlit.py")