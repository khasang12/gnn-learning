"""
Unit tests for data modules.
"""
import torch
import pytest
from torch_geometric.data import Data
from gnn_learning.data.generator import generate_synthetic_rec_data
from gnn_learning.data.loader import load_dataset


class TestSyntheticDataGenerator:
    """Test synthetic recommendation data generator."""

    def test_generate_synthetic_rec_data_basic(self):
        """Test basic data generation with default parameters."""
        data = generate_synthetic_rec_data(n_users=50, n_items=30, seed=42)

        assert isinstance(data, Data)
        assert hasattr(data, 'num_users')
        assert hasattr(data, 'num_items')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'test_edge_index')
        assert hasattr(data, 'train_edge_index')
        assert data.num_users == 50
        assert data.num_items == 30
        assert data.x.shape == (80, 16)  # n_users + n_items, 16 features

    def test_edge_generation(self):
        """Test that edges are generated correctly."""
        data = generate_synthetic_rec_data(n_users=10, n_items=5, edge_prob=0.5, seed=123)

        # Edge indices should be within valid ranges
        train_edges = data.edge_index
        test_edges = data.test_edge_index

        # All user indices < num_users
        assert torch.all(train_edges[0] < data.num_users)
        assert torch.all(test_edges[0] < data.num_users)

        # All item indices >= num_users and < num_users + num_items
        assert torch.all(train_edges[1] >= data.num_users)
        assert torch.all(train_edges[1] < data.num_users + data.num_items)
        assert torch.all(test_edges[1] >= data.num_users)
        assert torch.all(test_edges[1] < data.num_users + data.num_items)

        # Train and test edges should be disjoint
        train_set = set(tuple(pair) for pair in train_edges.t().tolist())
        test_set = set(tuple(pair) for pair in test_edges.t().tolist())
        assert len(train_set & test_set) == 0

    def test_reproducibility(self):
        """Test that same seed produces identical data."""
        data1 = generate_synthetic_rec_data(n_users=20, n_items=15, seed=42)
        data2 = generate_synthetic_rec_data(n_users=20, n_items=15, seed=42)

        # Compare tensors
        assert torch.equal(data1.x, data2.x)
        assert torch.equal(data1.edge_index, data2.edge_index)
        assert torch.equal(data1.test_edge_index, data2.test_edge_index)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data (likely)."""
        data1 = generate_synthetic_rec_data(n_users=20, n_items=15, seed=42)
        data2 = generate_synthetic_rec_data(n_users=20, n_items=15, seed=43)

        # At least some tensors should differ (not guaranteed but very likely)
        assert not torch.equal(data1.x, data2.x) or \
               not torch.equal(data1.edge_index, data2.edge_index)

    def test_minimum_edges_guarantee(self):
        """Test that generator creates at least some edges even with low probability."""
        # Extremely low probability should still yield edges due to minimum sampling
        data = generate_synthetic_rec_data(n_users=5, n_items=5, edge_prob=0.0001, seed=42)

        total_edges = data.edge_index.shape[1] + data.test_edge_index.shape[1]
        assert total_edges >= 1  # Should have at least one edge

    def test_train_test_split_ratio(self):
        """Test that train/test split respects approximate ratio."""
        n_users, n_items = 30, 20
        data = generate_synthetic_rec_data(n_users=n_users, n_items=n_items, train_ratio=0.8, seed=42)

        total_edges = data.edge_index.shape[1] + data.test_edge_index.shape[1]
        train_ratio = data.edge_index.shape[1] / total_edges

        # Allow small deviation due to rounding
        assert abs(train_ratio - 0.8) < 0.05


class TestDatasetLoader:
    """Test dataset loading functionality."""

    def test_load_dataset_cora(self, tmp_path):
        """Test loading Cora dataset (mocked or real)."""
        # This test may download dataset; we'll skip if not available
        # For CI, we could mock Planetoid
        try:
            dataset, data = load_dataset(name="Cora", root=str(tmp_path))
        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")

        assert dataset.num_classes == 7
        assert data.num_nodes == 2708
        assert data.num_edges == 10556  # With self-loops? Actually Cora has 5278 edges, but with transforms?
        assert data.num_node_features == 1433

        # Check masks exist
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')

    def test_load_dataset_citeseer(self, tmp_path):
        """Test loading CiteSeer dataset."""
        try:
            dataset, data = load_dataset(name="CiteSeer", root=str(tmp_path))
        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")

        assert dataset.num_classes == 6
        # Basic sanity checks
        assert data.num_nodes > 0
        assert data.num_edges > 0
        assert data.num_node_features > 0

    def test_load_dataset_pubmed(self, tmp_path):
        """Test loading PubMed dataset."""
        try:
            dataset, data = load_dataset(name="PubMed", root=str(tmp_path))
        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")

        assert dataset.num_classes == 3
        # Basic sanity checks
        assert data.num_nodes > 0
        assert data.num_edges > 0
        assert data.num_node_features > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])