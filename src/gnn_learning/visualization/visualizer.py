import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(model, data, epoch=None):
    """Create t-SNE visualization of node embeddings"""
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        # Assuming model has gcn1 layer or similar structure that outputs embeddings
        # For GCN model in this project
        if hasattr(model, 'gcn1'):
            embeddings = model.gcn1(x, edge_index).cpu().numpy()
        else:
            # Fallback or strict requirement
            # For LightGCN, it returns (user_emb, item_emb)
            # This function seems specific to GCN node classification
             embeddings = model.gcn1(x, edge_index).cpu().numpy()
        
        labels = data.y.cpu().numpy()

        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=labels, cmap='tab20', s=20, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label='Class')
        title = f't-SNE visualization of node embeddings'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        ax.set_title(title)
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig
