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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import time
import os
from PIL import Image

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
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
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

# Sidebar for controls
with st.sidebar:
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
        help="Number of hidden units in the GCN layer"
    )
    
    dropout_rate = st.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.8,
        value=0.5,
        step=0.1,
        help="Dropout probability for regularization"
    )
    
    # Training hyperparameters
    st.markdown("### Training Parameters")
    learning_rate = st.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.4f"
    )
    
    weight_decay = st.slider(
        "Weight Decay (L2)",
        min_value=0.0,
        max_value=0.01,
        value=0.0005,
        step=0.0001,
        format="%.4f"
    )
    
    num_epochs = st.slider(
        "Number of Epochs",
        min_value=10,
        max_value=500,
        value=200,
        step=10
    )
    
    # Action buttons
    st.markdown("### Actions")
    train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    visualize_button = st.button("📊 Visualize Embeddings", use_container_width=True)
    
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

# Define GCN model
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

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">📈 Training Progress</h2>', unsafe_allow_html=True)
    progress_placeholder = st.empty()
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()

with col2:
    st.markdown('<h2 class="sub-header">📊 Dataset Info</h2>', unsafe_allow_html=True)
    dataset_info_placeholder = st.empty()

# Initialize session state for storing results
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

# Load dataset function
@st.cache_resource
def load_dataset(name="Cora"):
    """Load and cache dataset"""
    dataset = Planetoid(root=f'data/{name}', name=name, transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data

# Train model function
def train_model(model, data, optimizer, criterion, num_epochs):
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

# Visualization function
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
                model, data, optimizer, criterion, num_epochs
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
        st.info("👈 Configure your model in the sidebar and click 'Train Model' to start training.")
    
    with metrics_placeholder.container():
        st.markdown("### 🎯 Expected Performance")
        st.write("With default settings, you can expect:")
        st.write("- **Test Accuracy**: ~80-85%")
        st.write("- **Training Time**: 1-2 minutes on CPU")
        st.write("- **Visualizations**: Loss curves and t-SNE embeddings")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Built with ❤️ for intermediate AI learners | 
    <a href="https://pytorch-geometric.readthedocs.io/" target="_blank">PyTorch Geometric</a> | 
    <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
    <p>Dataset: Cora citation network (2,708 scientific publications)</p>
</div>
""", unsafe_allow_html=True)