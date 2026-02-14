"""
Unit tests for GNN models.
"""
import torch
import pytest
from gnn_learning.models.gcn import GCN
from gnn_learning.models.lightgcn import LightGCN, bpr_loss


class TestGCN:
    """Test GCN model."""

    def test_initialization(self):
        """Test GCN initialization with different parameters."""
        model = GCN(num_features=10, hidden_dim=16, num_classes=3, dropout=0.5)
        assert model.gcn1.in_channels == 10
        assert model.gcn1.out_channels == 16
        assert model.gcn2.in_channels == 16
        assert model.gcn2.out_channels == 3
        assert model.dropout == 0.5

    def test_forward_pass(self, dummy_graph_data):
        """Test forward pass returns correct shape and log softmax."""
        data = dummy_graph_data
        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3)

        out = model(data)

        # Check output shape
        assert out.shape == (data.num_nodes, 3)
        # Check log softmax: sum exp should be close to 1 (log domain)
        assert torch.allclose(torch.exp(out).sum(dim=1), torch.ones(data.num_nodes), atol=1e-6)

    def test_training_mode(self, dummy_graph_data):
        """Test dropout behaves differently in train vs eval mode."""
        data = dummy_graph_data
        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3, dropout=0.5)

        model.train()
        out_train = model(data)
        model.eval()
        out_eval = model(data)

        # Outputs should differ due to dropout randomness
        # (not a strict guarantee but likely)
        assert not torch.allclose(out_train, out_eval, atol=1e-6)

    def test_parameter_gradients(self, dummy_graph_data):
        """Test gradients flow through parameters after backward pass."""
        data = dummy_graph_data
        model = GCN(num_features=data.num_node_features, hidden_dim=8, num_classes=3)
        criterion = torch.nn.NLLLoss()

        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # Check gradients exist for parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


class TestLightGCN:
    """Test LightGCN model and BPR loss."""

    def test_initialization(self):
        """Test LightGCN initialization."""
        model = LightGCN(num_users=100, num_items=50, embedding_dim=32, num_layers=3)

        assert model.num_users == 100
        assert model.num_items == 50
        assert model.embedding_dim == 32
        assert model.num_layers == 3
        assert model.user_embedding.weight.shape == (100, 32)
        assert model.item_embedding.weight.shape == (50, 32)

    def test_forward_pass(self, synthetic_rec_data):
        """Test forward pass returns correct embeddings."""
        data = synthetic_rec_data
        model = LightGCN(num_users=data.num_users, num_items=data.num_items, embedding_dim=16, num_layers=2)

        user_emb, item_emb = model(data)

        assert user_emb.shape == (data.num_users, 16)
        assert item_emb.shape == (data.num_items, 16)

    def test_predict_method(self, synthetic_rec_data):
        """Test predict method computes dot product scores."""
        data = synthetic_rec_data
        model = LightGCN(num_users=data.num_users, num_items=data.num_items, embedding_dim=8, num_layers=1)

        user_emb, item_emb = model(data)
        scores = model.predict(user_emb, item_emb)

        assert scores.shape == (data.num_users, data.num_items)
        # Check a few entries manually
        for i in range(min(3, data.num_users)):
            for j in range(min(3, data.num_items)):
                expected = torch.dot(user_emb[i], item_emb[j])
                assert torch.allclose(scores[i, j], expected, atol=1e-6)

    def test_bpr_loss(self):
        """Test BPR loss computation."""
        batch_size = 5
        embedding_dim = 10

        user_emb = torch.randn(batch_size, embedding_dim, requires_grad=True)
        pos_item_emb = torch.randn(batch_size, embedding_dim, requires_grad=True)
        neg_item_emb = torch.randn(batch_size, embedding_dim, requires_grad=True)

        loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        assert loss.shape == ()
        assert loss.requires_grad
        assert loss.item() > 0  # loss should be positive

        # Test gradient flow
        loss.backward()
        assert user_emb.grad is not None
        assert pos_item_emb.grad is not None
        assert neg_item_emb.grad is not None

    def test_bpr_loss_symmetry(self):
        """Test BPR loss decreases when positive score is higher."""
        embedding_dim = 5

        # Case 1: positive score much higher than negative
        user = torch.tensor([[1., 0., 0., 0., 0.]])
        pos = torch.tensor([[1., 0., 0., 0., 0.]])  # same as user
        neg = torch.tensor([[0., 1., 0., 0., 0.]])  # orthogonal
        loss1 = bpr_loss(user, pos, neg)

        # Case 2: positive score slightly higher
        user2 = torch.tensor([[1., 0., 0., 0., 0.]])
        pos2 = torch.tensor([[0.9, 0., 0., 0., 0.]])
        neg2 = torch.tensor([[0.8, 0., 0., 0., 0.]])
        loss2 = bpr_loss(user2, pos2, neg2)

        # Case 3: positive score lower (should give higher loss)
        user3 = torch.tensor([[1., 0., 0., 0., 0.]])
        pos3 = torch.tensor([[0.5, 0., 0., 0., 0.]])
        neg3 = torch.tensor([[0.9, 0., 0., 0., 0.]])
        loss3 = bpr_loss(user3, pos3, neg3)

        # loss1 should be low (positive clearly better)
        # loss3 should be high (negative better)
        # loss2 somewhere in between
        assert loss1 < loss3
        assert loss2 < loss3

    def test_device_handling(self, synthetic_rec_data):
        """Test model works on CPU device."""
        data = synthetic_rec_data
        model = LightGCN(num_users=data.num_users, num_items=data.num_items, embedding_dim=16, num_layers=2)

        # Ensure model parameters are on CPU
        assert model.user_embedding.weight.device == torch.device('cpu')
        assert model.item_embedding.weight.device == torch.device('cpu')

        user_emb, item_emb = model(data)
        assert user_emb.device == torch.device('cpu')
        assert item_emb.device == torch.device('cpu')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])