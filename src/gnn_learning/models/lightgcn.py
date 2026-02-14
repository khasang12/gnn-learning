import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class LightGCN(nn.Module):
    """
    Simplified LightGCN model for recommendation.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # User and item embeddings
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, data):
        # Get embeddings for all users and items
        user_emb = self.user_embedding(torch.arange(self.num_users, device=self.user_embedding.weight.device))
        item_emb = self.item_embedding(torch.arange(self.num_items, device=self.item_embedding.weight.device))

        # Concatenate to full node embeddings
        embeddings = torch.cat([user_emb, item_emb], dim=0)  # (num_users+num_items, dim)

        # LightGCN propagation: simple average over neighbors
        # We'll implement propagation using normalized adjacency matrix
        edge_index = data.edge_index
        num_nodes = self.num_users + self.num_items

        # Build symmetric adjacency matrix with self-loops and normalize
        from torch_geometric.utils import add_self_loops, degree
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=embeddings.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate through layers
        all_embeddings = [embeddings]
        for _ in range(self.num_layers):
            # Message passing: sum_{j in N(i)} e_j / sqrt(d_i*d_j)
            messages = embeddings[col] * norm.view(-1, 1)
            embeddings = scatter(messages, row, dim=0, dim_size=num_nodes, reduce='sum')
            all_embeddings.append(embeddings)

        # Final embeddings: average of all layers
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        final_user_emb = final_embeddings[:self.num_users]
        final_item_emb = final_embeddings[self.num_users:]

        return final_user_emb, final_item_emb

    def predict(self, user_emb, item_emb):
        """Compute dot product scores between user and item embeddings."""
        return torch.matmul(user_emb, item_emb.t())


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """
    Bayesian Personalized Ranking loss.
    user_emb: (batch_size, dim)
    pos_item_emb: (batch_size, dim)
    neg_item_emb: (batch_size, dim)
    """
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    return loss
