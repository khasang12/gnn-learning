#!/usr/bin/env python3
"""
Interactive GNN Example with Streamlit
A web-based interface for experimenting with Graph Neural Networks
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import scatter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random

# Set page config
st.set_page_config(
    page_title="GNN Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    :root {
        --primary-color: #3B82F6;
        --primary-dark: #1E3A8A;
        --secondary-color: #8B5CF6;
        --background-light: #F8FAFC;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --text-dark: #1F2937;
        --text-light: #6B7280;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.15);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --border-radius: 0.75rem;
    }

    .main-header {
        font-size: 2.75rem;
        font-weight: 800;
        background: var(--background-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }

    .sub-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary-dark);
        margin-top: 2rem;
        margin-bottom: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-color);
        display: inline-block;
    }

    .info-box {
        background: linear-gradient(145deg, #ffffff, #f1f5f9);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border-left: 6px solid var(--primary-color);
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
        transition: var(--transition);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }

    .info-box:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
        border-left: 6px solid var(--secondary-color);
    }

    .info-box strong {
        color: var(--primary-dark);
        font-size: 1.25rem;
        font-weight: 700;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }

    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-3px);
        border-color: var(--primary-color);
    }

    .metric-card .stMetric {
        font-family: 'Inter', sans-serif;
    }

    .metric-card .stMetric label {
        color: var(--text-light);
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .stMetric div {
        color: var(--primary-dark);
        font-weight: 800;
        font-size: 2rem;
    }

    /* Button enhancements */
    .stButton > button {
        border-radius: 0.75rem;
        font-weight: 600;
        transition: var(--transition);
        border: none;
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: var(--background-gradient);
        border-radius: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: var(--transition);
    }

    .stTabs [aria-selected="true"] {
        background: var(--background-gradient) !important;
        color: white !important;
        box-shadow: var(--shadow-md);
    }

    /* Improve general typography */
    body {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        color: var(--text-dark);
        line-height: 1.6;
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        line-height: 1.2;
    }

    /* Sidebar enhancements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }

    /* Responsive improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 Graph Neural Network Learning Lab</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>Interactive GNN Example for Intermediate AI Learners</strong><br>
    This interactive demo lets you experiment with Graph Convolutional Networks (GCNs)
    for node classification on the Cora citation network dataset. Adjust hyperparameters,
    visualize training progress, and explore node embeddings in real-time.
</div>
""", unsafe_allow_html=True)

# Define GCN model (used in GNN Basics tab)
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize session state for storing results (shared across tabs)
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'train_losses' not in st.session_state:
    st.session_state.train_losses = []
if 'val_losses' not in st.session_state:
    st.session_state.val_losses = []
if 'train_accs' not in st.session_state:
    st.session_state.train_accs = []
if 'val_accs' not in st.session_state:
    st.session_state.val_accs = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = None

# Recommendation system session state
if 'rec_model' not in st.session_state:
    st.session_state.rec_model = None
if 'rec_data' not in st.session_state:
    st.session_state.rec_data = None
if 'rec_train_losses' not in st.session_state:
    st.session_state.rec_train_losses = []
if 'rec_val_losses' not in st.session_state:
    st.session_state.rec_val_losses = []
if 'rec_metrics' not in st.session_state:
    st.session_state.rec_metrics = None

# Load dataset function (cached)
@st.cache_resource
def load_dataset(name="Cora"):
    """Load and cache dataset"""
    dataset = Planetoid(root=f'data/{name}', name=name, transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data

# Train model function for GNN Basics
def train_model(model, data, optimizer, criterion, num_epochs, progress_placeholder):
    """Train the model with progress tracking"""
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    progress_bar = progress_placeholder.progress(0)
    status_text = progress_placeholder.empty()

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Training accuracy
        pred = out[data.train_mask].argmax(dim=1)
        train_acc = (pred == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data)
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
            pred = out[data.val_mask].argmax(dim=1)
            val_acc = (pred == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update progress
        progress = epoch / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch}/{num_epochs} - Train Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    progress_bar.empty()
    status_text.empty()

    return train_losses, val_losses, train_accs, val_accs

# Visualization function for embeddings
def visualize_embeddings(model, data, epoch=None):
    """Create t-SNE visualization of node embeddings"""
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
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

# ========== Recommendation System Core Functions ==========
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
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=train_edge_index)
    data.num_users = n_users
    data.num_items = n_items
    data.test_edge_index = test_edge_index
    data.train_edge_index = train_edge_index

    return data


class LightGCN(torch.nn.Module):
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


def train_lightgcn(model, data, optimizer, num_epochs=100, batch_size=1024):
    """
    Train LightGCN model using BPR loss with negative sampling.
    """
    train_losses = []
    model.train()

    # Get edge list for training
    edge_index = data.edge_index.cpu().numpy()
    train_edges = list(zip(edge_index[0], edge_index[1]))

    for epoch in range(1, num_epochs + 1):
        # Shuffle edges
        np.random.shuffle(train_edges)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_edges), batch_size):
            batch_edges = train_edges[i:i + batch_size]
            if len(batch_edges) < 2:
                continue

            # Prepare batch: positive pairs (user, item)
            users = [u for u, _ in batch_edges]
            pos_items = [i - data.num_users for _, i in batch_edges]  # convert to item index

            # Negative sampling: random items not interacted with
            neg_items = np.random.randint(0, data.num_items, len(batch_edges))

            # Convert to tensors
            users = torch.tensor(users, dtype=torch.long, device=data.x.device)
            pos_items = torch.tensor(pos_items, dtype=torch.long, device=data.x.device)
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=data.x.device)

            # Get embeddings
            user_emb, item_emb = model(data)
            user_emb_batch = user_emb[users]
            pos_item_emb_batch = item_emb[pos_items]
            neg_item_emb_batch = item_emb[neg_items]

            # Compute loss
            loss = bpr_loss(user_emb_batch, pos_item_emb_batch, neg_item_emb_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return train_losses


def evaluate_rec_system(model, data, k=10):
    """
    Evaluate recommendation system using Recall@K and NDCG@K.
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(data)

        # Compute scores for all user-item pairs
        scores = model.predict(user_emb, item_emb)  # (num_users, num_items)

        # Get test edges
        test_edge_index = data.test_edge_index.cpu().numpy()
        test_edges = list(zip(test_edge_index[0], test_edge_index[1]))

        recall_sum = 0.0
        ndcg_sum = 0.0
        num_users_evaluated = 0

        # For each user with test interactions
        test_users = set([u for u, _ in test_edges])
        for user in test_users:
            # Get test items for this user
            user_test_items = [i - data.num_users for u, i in test_edges if u == user]
            if not user_test_items:
                continue

            # Get predicted scores for this user
            user_scores = scores[user].cpu().numpy()

            # Rank items by score (higher is better)
            top_k_indices = np.argsort(user_scores)[-k:][::-1]

            # Compute Recall@K
            hits = len(set(top_k_indices) & set(user_test_items))
            recall = hits / len(user_test_items)
            recall_sum += recall

            # Compute NDCG@K
            dcg = 0.0
            for idx, item in enumerate(top_k_indices):
                if item in user_test_items:
                    dcg += 1 / np.log2(idx + 2)  # idx starts from 0
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(user_test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_sum += ndcg

            num_users_evaluated += 1

        avg_recall = recall_sum / num_users_evaluated if num_users_evaluated > 0 else 0
        avg_ndcg = ndcg_sum / num_users_evaluated if num_users_evaluated > 0 else 0

        return avg_recall, avg_ndcg


# ========== GNN Basics Tab ==========
def show_gnn_basics_tab():
    """Display the original GNN node classification interface"""

    # Two-column layout: left for controls, right for dataset info
    col_controls, col_main = st.columns([1, 2])

    with col_controls:
        st.markdown("## ⚙️ Model Configuration")

        # Dataset selection
        dataset_name = st.selectbox(
            "Dataset",
            ["Cora", "CiteSeer", "PubMed"],
            help="Select which citation network dataset to use"
        )

        # Model hyperparameters
        st.markdown("### Model Architecture")
        hidden_dim = st.slider(
            "Hidden Dimension Size",
            min_value=8,
            max_value=128,
            value=16,
            step=8,
            help="Number of hidden units in the GCN layer",
            key="hidden_dim_slider_basics"
        )

        dropout_rate = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.8,
            value=0.5,
            step=0.1,
            help="Dropout probability for regularization",
            key="dropout_slider_basics"
        )

        # Training hyperparameters
        st.markdown("### Training Parameters")
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.4f",
            key="learning_rate_slider_basics"
        )

        weight_decay = st.slider(
            "Weight Decay (L2)",
            min_value=0.0,
            max_value=0.01,
            value=0.0005,
            step=0.0001,
            format="%.4f",
            key="weight_decay_slider_basics"
        )

        num_epochs = st.slider(
            "Number of Epochs",
            min_value=10,
            max_value=500,
            value=200,
            step=10,
            key="num_epochs_slider_basics"
        )

        # Action buttons
        st.markdown("### Actions")
        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True, key="train_button_basics")
        visualize_button = st.button("📊 Visualize Embeddings", use_container_width=True, key="visualize_button_basics")

        # Information
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This demo uses:
        - **PyTorch Geometric** for graph operations
        - **Streamlit** for interactive UI
        - **Cora Dataset**: 2,708 papers, 5,429 citations
        - **Task**: Classify papers into 7 categories
        """)

    with col_main:
        # Sub-columns for training progress and dataset info
        col_progress, col_info = st.columns([2, 1])

        with col_progress:
            st.markdown('<h2 class="sub-header">📈 Training Progress</h2>', unsafe_allow_html=True)
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

        with col_info:
            st.markdown('<h2 class="sub-header">📊 Dataset Info</h2>', unsafe_allow_html=True)
            dataset_info_placeholder = st.empty()

    # Load dataset on startup
    if st.session_state.dataset is None:
        with st.spinner("Loading dataset..."):
            dataset, data = load_dataset(dataset_name)
            st.session_state.dataset = dataset
            st.session_state.data = data

            # Display dataset info
            with dataset_info_placeholder.container():
                st.metric("Nodes", data.num_nodes)
                st.metric("Edges", data.num_edges)
                st.metric("Features", data.num_node_features)
                st.metric("Classes", dataset.num_classes)
                st.metric("Training Nodes", data.train_mask.sum().item())
                st.metric("Test Nodes", data.test_mask.sum().item())

    # Handle training button
    if train_button:
        if st.session_state.data is None:
            st.error("Please load dataset first")
        else:
            # Clear previous results
            st.session_state.train_losses = []
            st.session_state.val_losses = []
            st.session_state.train_accs = []
            st.session_state.val_accs = []

            # Create model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = st.session_state.data.to(device)
            dataset = st.session_state.dataset

            model = GCN(dataset.num_node_features, hidden_dim, dataset.num_classes, dropout_rate).to(device)
            st.session_state.model = model

            # Setup optimizer and criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.NLLLoss()

            # Train model
            with st.spinner("Training GNN model..."):
                train_losses, val_losses, train_accs, val_accs = train_model(
                    model, data, optimizer, criterion, num_epochs, progress_placeholder
                )

                st.session_state.train_losses = train_losses
                st.session_state.val_losses = val_losses
                st.session_state.train_accs = train_accs
                st.session_state.val_accs = val_accs

                # Final evaluation
                model.eval()
                with torch.no_grad():
                    out = model(data)
                    test_loss = criterion(out[data.test_mask], data.y[data.test_mask]).item()
                    pred = out[data.test_mask].argmax(dim=1)
                    test_acc = (pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

                    st.session_state.test_results = {
                        'test_loss': test_loss,
                        'test_acc': test_acc
                    }

            # Display training plots
            if train_losses and val_losses:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                ax1.plot(train_losses, label='Training Loss', linewidth=2)
                ax1.plot(val_losses, label='Validation Loss', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
                ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training and Validation Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close()

                # Display metrics
                if st.session_state.test_results:
                    test_loss = st.session_state.test_results['test_loss']
                    test_acc = st.session_state.test_results['test_acc']

                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                        with col2:
                            st.metric("Final Val Loss", f"{val_losses[-1]:.4f}")
                        with col3:
                            st.metric("Test Loss", f"{test_loss:.4f}")
                        with col4:
                            st.metric("Test Accuracy", f"{test_acc:.4f}")

                        st.success(f"✅ Training completed! Model achieved {test_acc:.2%} accuracy on test set.")

    # Handle visualization button
    if visualize_button:
        if st.session_state.model is None:
            st.warning("Please train a model first")
        else:
            with st.spinner("Generating t-SNE visualization..."):
                fig = visualize_embeddings(st.session_state.model, st.session_state.data)
                chart_placeholder.pyplot(fig)
                plt.close()

    # Display instructions if no action taken
    if not train_button and not visualize_button:
        with chart_placeholder.container():
            st.info("👈 Configure your model on the left and click 'Train Model' to start training.")

        with metrics_placeholder.container():
            st.markdown("### 🎯 Expected Performance")
            st.write("With default settings, you can expect:")
            st.write("- **Test Accuracy**: ~80-85%")
            st.write("- **Training Time**: 1-2 minutes on CPU")
            st.write("- **Visualizations**: Loss curves and t-SNE embeddings")

# ========== GNN for Recommendations Tab ==========
def show_recommendation_tab():
    """Display the GNN recommendation system interface"""
    st.markdown('<h2 class="sub-header">🎯 GNN for Recommendation Systems</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Interactive Demo: GNNs for Recommendation Systems</strong><br>
        This demo shows how Graph Neural Networks can be used for recommendation tasks.
        We create a synthetic user-item interaction graph and train a LightGCN model
        to predict unseen interactions.
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout: left for controls, right for main area
    col_controls, col_main = st.columns([1, 2])

    with col_controls:
        st.markdown("## ⚙️ Configuration")

        # Dataset parameters
        st.markdown("### Synthetic Dataset")
        n_users = st.slider("Number of Users", min_value=10, max_value=200, value=50, step=10, key="n_users_slider_rec")
        n_items = st.slider("Number of Items", min_value=10, max_value=200, value=30, step=10, key="n_items_slider_rec")
        edge_prob = st.slider("Edge Probability", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="edge_prob_slider_rec")
        train_ratio = st.slider("Train Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05, key="train_ratio_slider_rec")

        # Model hyperparameters
        st.markdown("### Model Architecture")
        embedding_dim = st.slider("Embedding Dimension", min_value=8, max_value=128, value=64, step=8, key="embedding_dim_slider_rec")
        num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=3, key="num_layers_slider_rec")

        # Training hyperparameters
        st.markdown("### Training Parameters")
        learning_rate = st.slider("Learning Rate", min_value=1e-4, max_value=1e-2, value=1e-3, step=1e-4, format="%.4f", key="learning_rate_slider_rec")
        num_epochs = st.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10, key="num_epochs_slider_rec")
        batch_size = st.slider("Batch Size", min_value=32, max_value=1024, value=256, step=32, key="batch_size_slider_rec")

        # Action buttons
        st.markdown("### Actions")
        generate_button = st.button("📊 Generate Dataset", type="primary", use_container_width=True, key="generate_button_rec")
        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True, key="train_button_rec")
        evaluate_button = st.button("📈 Evaluate Model", use_container_width=True, key="evaluate_button_rec")

        # Information
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This demo uses:
        - **LightGCN**: Simplified GNN for recommendation
        - **Synthetic Data**: User-item bipartite graph
        - **Evaluation**: Recall@K and NDCG@K metrics
        """)

    with col_main:
        # Sub-columns for dataset info and training progress
        col_info, col_progress = st.columns([1, 1])

        with col_info:
            st.markdown('<h3 class="sub-header">📊 Dataset Info</h3>', unsafe_allow_html=True)
            dataset_info_placeholder = st.empty()

        with col_progress:
            st.markdown('<h3 class="sub-header">📈 Training Progress</h3>', unsafe_allow_html=True)
            progress_placeholder = st.empty()

        # Placeholders for metrics and visualization
        st.markdown("### 📋 Evaluation Metrics")
        metrics_placeholder = st.empty()

        st.markdown("### 📉 Training Loss Curve")
        chart_placeholder = st.empty()

    # Generate dataset button handler
    if generate_button:
        with st.spinner("Generating synthetic dataset..."):
            data = generate_synthetic_rec_data(
                n_users=n_users, n_items=n_items,
                edge_prob=edge_prob, train_ratio=train_ratio
            )
            st.session_state.rec_data = data
            st.session_state.rec_n_users = n_users
            st.session_state.rec_n_items = n_items
            st.session_state.rec_model = None
            st.session_state.rec_train_losses = []
            st.session_state.rec_metrics = None

            # Display dataset info
            with dataset_info_placeholder.container():
                st.metric("Users", n_users)
                st.metric("Items", n_items)
                st.metric("Training Edges", data.edge_index.shape[1])
                st.metric("Test Edges", data.test_edge_index.shape[1])
                st.metric("Edge Density", f"{(data.edge_index.shape[1] + data.test_edge_index.shape[1]) / (n_users * n_items):.2%}")

            st.success(f"✅ Generated dataset with {n_users} users, {n_items} items, and {data.edge_index.shape[1] + data.test_edge_index.shape[1]} total edges.")

    # Train model button handler
    if train_button:
        if st.session_state.rec_data is None:
            st.error("Please generate dataset first")
        else:
            # Clear previous results
            st.session_state.rec_train_losses = []
            st.session_state.rec_metrics = None

            # Create model - use stored values if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = st.session_state.rec_data.to(device)

            # Use stored values from dataset generation
            stored_n_users = st.session_state.get('rec_n_users', n_users)
            stored_n_items = st.session_state.get('rec_n_items', n_items)

            model = LightGCN(
                num_users=stored_n_users,
                num_items=stored_n_items,
                embedding_dim=embedding_dim,
                num_layers=num_layers
            ).to(device)
            st.session_state.rec_model = model

            # Setup optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train model with progress tracking
            progress_bar = progress_placeholder.progress(0)
            status_text = progress_placeholder.empty()

            train_losses = []
            for epoch in range(1, num_epochs + 1):
                # Train for one epoch (simplified)
                model.train()
                edge_index = data.edge_index.cpu().numpy()
                train_edges = list(zip(edge_index[0], edge_index[1]))
                np.random.shuffle(train_edges)

                total_loss = 0.0
                num_batches = 0

                for i in range(0, len(train_edges), batch_size):
                    batch_edges = train_edges[i:i + batch_size]
                    if len(batch_edges) < 2:
                        continue

                    users = [u for u, _ in batch_edges]
                    pos_items = [i - stored_n_users for _, i in batch_edges]
                    neg_items = np.random.randint(0, stored_n_items, len(batch_edges))

                    users = torch.tensor(users, dtype=torch.long, device=device)
                    pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
                    neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)

                    user_emb, item_emb = model(data)
                    user_emb_batch = user_emb[users]
                    pos_item_emb_batch = item_emb[pos_items]
                    neg_item_emb_batch = item_emb[neg_items]

                    loss = bpr_loss(user_emb_batch, pos_item_emb_batch, neg_item_emb_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                train_losses.append(avg_loss)

                # Update progress
                progress = epoch / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

            progress_bar.empty()
            status_text.empty()
            st.session_state.rec_train_losses = train_losses

            # Plot training loss
            if train_losses:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(train_losses, label='Training Loss', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close()

            st.success(f"✅ Training completed! Final loss: {train_losses[-1]:.4f}")

    # Evaluate model button handler
    if evaluate_button:
        if st.session_state.rec_model is None:
            st.warning("Please train a model first")
        elif st.session_state.rec_data is None:
            st.error("Please generate dataset first")
        else:
            with st.spinner("Evaluating model..."):
                model = st.session_state.rec_model
                data = st.session_state.rec_data
                recall, ndcg = evaluate_rec_system(model, data, k=10)

                st.session_state.rec_metrics = {'recall': recall, 'ndcg': ndcg}

                with metrics_placeholder.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Recall@10", f"{recall:.4f}")
                    with col2:
                        st.metric("NDCG@10", f"{ndcg:.4f}")

                    st.success(f"✅ Evaluation completed! Model achieves Recall@10={recall:.2%}, NDCG@10={ndcg:.2%}")

    # Display instructions if no action taken
    if not generate_button and not train_button and not evaluate_button:
        with dataset_info_placeholder.container():
            st.info("👈 Configure parameters on the left and click 'Generate Dataset' to start.")

        with metrics_placeholder.container():
            st.markdown("### 🎯 Expected Performance")
            st.write("With default settings, you can expect:")
            st.write("- **Recall@10**: ~0.2-0.4 (higher is better)")
            st.write("- **NDCG@10**: ~0.1-0.3 (higher is better)")
            st.write("- **Training Time**: 30-60 seconds on CPU")

        with chart_placeholder.container():
            st.info("Training loss curve will appear here after training.")

# ========== Main App with Tabs ==========
tab_basics, tab_recommendation = st.tabs(["🧠 GNN Basics", "🎯 GNN for Recommendations"])

with tab_basics:
    show_gnn_basics_tab()

with tab_recommendation:
    show_recommendation_tab()

# Footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center;
    color: #6B7280;
    font-size: 0.9rem;
    padding: 1.5rem;
    background: linear-gradient(145deg, #f8fafc, #f1f5f9);
    border-radius: 0.75rem;
    margin-top: 2rem;
    border: 1px solid rgba(59, 130, 246, 0.1);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
">
    <p style="
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    ">
        🧠 Graph Neural Network Learning Lab
    </p>
    <p style="margin-bottom: 1rem;">
        Built with ❤️ for intermediate AI learners |
        <a href="https://pytorch-geometric.readthedocs.io/" target="_blank" style="color: #3B82F6; text-decoration: none; font-weight: 600;">PyTorch Geometric</a> |
        <a href="https://streamlit.io/" target="_blank" style="color: #3B82F6; text-decoration: none; font-weight: 600;">Streamlit</a>
    </p>
    <p style="
        font-size: 0.85rem;
        color: #6B7280;
        opacity: 0.8;
    ">
        <strong>Demo 1:</strong> GCN for Node Classification (Cora dataset) •
        <strong>Demo 2:</strong> LightGCN for Recommendation Systems (Synthetic data)
    </p>
</div>
""", unsafe_allow_html=True)