"""
Unit tests for training modules.
"""
import torch
import pytest
import numpy as np
from unittest.mock import Mock, patch
from gnn_learning.training.trainer import train_model, train_lightgcn
from gnn_learning.models.gcn import GCN
from gnn_learning.models.lightgcn import LightGCN


class TestTrainModel:
    """Test GCN training function."""

    def test_train_model_basic(self, dummy_graph_data):
        """Test training loop runs without errors and returns metrics."""
        data = dummy_graph_data
        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()

        num_epochs = 5
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, data, optimizer, criterion, num_epochs
        )

        # Check returned lists have correct length
        assert len(train_losses) == num_epochs
        assert len(val_losses) == num_epochs
        assert len(train_accs) == num_epochs
        assert len(val_accs) == num_epochs

        # Losses should be floats
        assert all(isinstance(loss, float) for loss in train_losses)
        assert all(isinstance(loss, float) for loss in val_losses)

        # Accuracies should be between 0 and 1
        assert all(0 <= acc <= 1 for acc in train_accs)
        assert all(0 <= acc <= 1 for acc in val_accs)

        # Training loss should decrease over epochs (not guaranteed but typical)
        # We'll just check that the function ran without error

    def test_train_model_with_progress_callback(self, dummy_graph_data):
        """Test training with progress callback."""
        data = dummy_graph_data
        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()

        mock_callback = Mock()
        num_epochs = 3

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, data, optimizer, criterion, num_epochs, mock_callback
        )

        # Callback should have been called for each epoch
        assert mock_callback.call_count == num_epochs

        # Check callback arguments
        for i, call in enumerate(mock_callback.call_args_list):
            args, kwargs = call
            assert len(args) == 2
            progress, status_text = args
            # Progress should increase each epoch
            expected_progress = (i + 1) / num_epochs
            assert abs(progress - expected_progress) < 1e-6
            assert isinstance(status_text, str)
            assert "Epoch" in status_text

    def test_train_model_device_handling(self, dummy_graph_data):
        """Test training works on CPU (default)."""
        data = dummy_graph_data
        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()

        # Ensure model and data are on CPU
        assert next(model.parameters()).device == torch.device('cpu')
        assert data.x.device == torch.device('cpu')

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, data, optimizer, criterion, num_epochs=2
        )

        # Should complete without error
        assert len(train_losses) == 2

    def test_train_model_validation_masks(self, dummy_graph_data):
        """Test that validation uses val_mask."""
        data = dummy_graph_data
        # Ensure val_mask has some nodes
        assert data.val_mask.sum().item() > 0

        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, data, optimizer, criterion, num_epochs=2
        )

        # Validation losses should be computed
        assert all(loss > 0 for loss in val_losses)


class TestTrainLightGCN:
    """Test LightGCN training function."""

    def test_train_lightgcn_basic(self, synthetic_rec_data):
        """Test LightGCN training runs without errors."""
        data = synthetic_rec_data
        model = LightGCN(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=16,
            num_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        num_epochs = 3
        train_losses = train_lightgcn(model, data, optimizer, num_epochs=num_epochs)

        assert len(train_losses) == num_epochs
        assert all(isinstance(loss, float) for loss in train_losses)
        assert all(loss > 0 for loss in train_losses)  # BPR loss positive

    def test_train_lightgcn_with_progress_callback(self, synthetic_rec_data):
        """Test LightGCN training with progress callback."""
        data = synthetic_rec_data
        model = LightGCN(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=16,
            num_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        mock_callback = Mock()
        num_epochs = 2

        train_losses = train_lightgcn(
            model, data, optimizer, num_epochs=num_epochs, progress_callback=mock_callback
        )

        # Callback should have been called for each epoch
        assert mock_callback.call_count == num_epochs

        for i, call in enumerate(mock_callback.call_args_list):
            args, kwargs = call
            assert len(args) == 2
            progress, status_text = args
            expected_progress = (i + 1) / num_epochs
            assert abs(progress - expected_progress) < 1e-6
            assert isinstance(status_text, str)
            assert "Epoch" in status_text

    def test_train_lightgcn_batch_size_handling(self, synthetic_rec_data):
        """Test training with different batch sizes."""
        data = synthetic_rec_data
        model = LightGCN(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=8,
            num_layers=1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Test with batch size larger than number of edges
        train_losses = train_lightgcn(
            model, data, optimizer, num_epochs=2, batch_size=10000
        )

        assert len(train_losses) == 2

        # Test with small batch size
        train_losses = train_lightgcn(
            model, data, optimizer, num_epochs=2, batch_size=2
        )

        assert len(train_losses) == 2

    def test_train_lightgcn_negative_sampling(self, synthetic_rec_data):
        """Test that negative sampling works correctly."""
        data = synthetic_rec_data
        model = LightGCN(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=8,
            num_layers=1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Patch random.randint to verify negative items are within range
        with patch('numpy.random.randint') as mock_randint:
            mock_randint.return_value = np.zeros(data.edge_index.shape[1], dtype=int)
            train_losses = train_lightgcn(model, data, optimizer, num_epochs=1, batch_size=32)

            # Check randint was called with correct bounds
            mock_randint.assert_called_with(0, data.num_items, mock_randint.call_args[0][2])

    def test_train_lightgcn_edge_conversion(self, synthetic_rec_data):
        """Test edge index conversion to Python list works."""
        data = synthetic_rec_data
        model = LightGCN(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=8,
            num_layers=1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_losses = train_lightgcn(model, data, optimizer, num_epochs=1)

        # Should complete without error
        assert len(train_losses) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])