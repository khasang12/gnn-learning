#!/usr/bin/env python3
"""
Test the recommendation system core functions using the new package structure.
"""

import torch
import numpy as np
import random
import sys
import unittest
from torch_geometric.data import Data

# Import from package
try:
    from gnn_learning.data.generator import generate_synthetic_rec_data
    from gnn_learning.models.lightgcn import LightGCN, bpr_loss
except ImportError as e:
    print(f"✗ Failed to import from gnn_learning package: {e}")
    sys.exit(1)

print("Testing recommendation system core functions...")

class TestRecommendationCore(unittest.TestCase):
    def test_synthetic_data_generation(self):
        print("\n1. Testing synthetic data generation...")
        data = generate_synthetic_rec_data(n_users=50, n_items=30, seed=42)
        
        print(f"✓ Generated synthetic data with {data.num_users} users, {data.num_items} items")
        print(f"✓ Training edges: {data.edge_index.shape[1]}")
        print(f"✓ Test edges: {data.test_edge_index.shape[1]}")
        
        self.assertEqual(data.num_users, 50)
        self.assertEqual(data.num_items, 30)

    def test_lightgcn_model(self):
        print("\n2. Testing LightGCN model...")
        data = generate_synthetic_rec_data(n_users=50, n_items=30, seed=42)
        
        try:
            model = LightGCN(data.num_users, data.num_items, embedding_dim=32, num_layers=2)
            user_emb, item_emb = model(data)
            
            print(f"✓ Created LightGCN model with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"✓ User embeddings shape: {user_emb.shape}")
            print(f"✓ Item embeddings shape: {item_emb.shape}")
            
            self.assertEqual(user_emb.shape, (50, 32))
            self.assertEqual(item_emb.shape, (30, 32))
            
        except Exception as e:
            print(f"✗ Error testing LightGCN: {e}")
            raise e

    def test_bpr_loss(self):
        print("\n3. Testing BPR loss...")
        # Create dummy embeddings
        batch_size = 10
        embedding_dim = 32
        user_emb_dummy = torch.randn(batch_size, embedding_dim, requires_grad=True)
        pos_item_emb_dummy = torch.randn(batch_size, embedding_dim, requires_grad=True)
        neg_item_emb_dummy = torch.randn(batch_size, embedding_dim, requires_grad=True)

        loss = bpr_loss(user_emb_dummy, pos_item_emb_dummy, neg_item_emb_dummy)
        print(f"✓ BPR loss computed: {loss.item():.4f}")
        
        self.assertTrue(loss.requires_grad)

if __name__ == '__main__':
    unittest.main()