#!/usr/bin/env python3
"""
Interactive GNN Example with Streamlit
A web-based interface for experimenting with Graph Neural Networks
"""

import sys
import os

# Import path setup utility
try:
    import utils
    utils.setup_paths()
except ImportError:
    # Fallback if utils cannot be imported (should not happen if in same dir)
    # But for safety, we try to add current dir to path if strictly needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import utils
    utils.setup_paths()

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import from the new package
from gnn_learning.models.gcn import GCN
from gnn_learning.models.lightgcn import LightGCN
from gnn_learning.data.loader import load_dataset
from gnn_learning.data.generator import generate_synthetic_rec_data
from gnn_learning.training.trainer import train_model, train_lightgcn
from gnn_learning.evaluation.evaluator import evaluate_rec_system
from gnn_learning.visualization.visualizer import visualize_embeddings

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

# Initialize session state
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

# Wrapper for progress callback
class ProgressWrapper:
    def __init__(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text

    def __call__(self, progress, text):
        self.progress_bar.progress(progress)
        self.status_text.text(text)

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
            progress_container = st.container()
            results_container = st.container()
            
            with progress_container:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
            
            with results_container:
                chart_placeholder = st.empty()
                metrics_placeholder = st.empty()

        with col_info:
            st.markdown('<h2 class="sub-header">📊 Dataset Info</h2>', unsafe_allow_html=True)
            dataset_info_placeholder = st.empty()

    # Load dataset on startup
    if st.session_state.dataset is None:
        with st.spinner("Loading dataset..."):
            # Note: load_dataset signature might need adjustment if I changed it
            # My extracted load_dataset takes name and root. Default root is 'data'.
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
                progress_bar = progress_placeholder.progress(0)
                status_text = status_placeholder.empty()
                progress_callback = ProgressWrapper(progress_bar, status_text)
                
                train_losses, val_losses, train_accs, val_accs = train_model(
                    model, data, optimizer, criterion, num_epochs, progress_callback
                )

                st.session_state.train_losses = train_losses
                st.session_state.val_losses = val_losses
                st.session_state.train_accs = train_accs
                st.session_state.val_accs = val_accs
                
                # Clear progress bar after detailed training finishes
                progress_bar.empty()
                status_text.empty()

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
            st.toast("Generating visualization...")
            with chart_placeholder:
                with st.spinner("Generating t-SNE visualization..."):
                    fig = visualize_embeddings(st.session_state.model, st.session_state.data)
                    st.pyplot(fig)
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
    
    col_controls, col_main = st.columns([1, 2])
    
    with col_controls:
        st.markdown("### Data & Model Settings")
        
        n_users = st.slider("Number of Users", 50, 500, 100, 50, key="rec_n_users")
        n_items = st.slider("Number of Items", 20, 200, 50, 10, key="rec_n_items")
        embedding_dim = st.slider("Embedding Dimension", 8, 64, 32, 8, key="rec_emb_dim")
        num_layers = st.slider("GNN Layers", 1, 5, 3, 1, key="rec_layers")
        
        st.markdown("### Training Settings")
        rec_epochs = st.slider("Epochs", 50, 500, 200, 50, key="rec_epochs")
        rec_lr = st.number_input("Learning Rate", 0.001, 0.1, 0.01, 0.001, format="%.4f", key="rec_lr")
        
        gen_data_btn = st.button("🎲 Generate New Data", use_container_width=True)
        train_rec_btn = st.button("🚀 Train Recommender", type="primary", use_container_width=True)
        
    with col_main:
        results_container = st.container()
        
    # Generate data
    if gen_data_btn:
        with st.spinner("Generating synthetic interaction data..."):
            st.session_state.rec_data = generate_synthetic_rec_data(n_users=n_users, n_items=n_items)
            st.toast(f"Generated data with {n_users} users and {n_items} items")
            
    # Initial data load if none exists
    if st.session_state.rec_data is None:
        with st.spinner("Initializing default data..."):
            st.session_state.rec_data = generate_synthetic_rec_data(n_users=n_users, n_items=n_items)

            
    # Display data stats
    if st.session_state.rec_data is not None:
        data = st.session_state.rec_data
        with results_container:
            st.info(f"**Data Stats**: {data.num_users} Users, {data.num_items} Items, "
                    f"{data.edge_index.shape[1]} Interactions (Train), {data.test_edge_index.shape[1]} (Test)")
            
    # Train
    if train_rec_btn:
        if st.session_state.rec_data is None:
            st.error("Please generate data first")
        else:
            data = st.session_state.rec_data
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device)
            
            # Initialize model
            model = LightGCN(
                num_users=data.num_users,
                num_items=data.num_items,
                embedding_dim=embedding_dim,
                num_layers=num_layers
            ).to(device)
            st.session_state.rec_model = model
            
            optimizer = torch.optim.Adam(model.parameters(), lr=rec_lr)
            
            # Progress bar
            progress_bar = results_container.progress(0)
            status_text = results_container.empty()
            
            # Train
            start_time = time.time()
            progress_callback = ProgressWrapper(progress_bar, status_text)
            
            train_losses = train_lightgcn(
                model, data, optimizer, num_epochs=rec_epochs, progress_callback=progress_callback
            )
            
            st.session_state.rec_train_losses = train_losses
            
            # Evaluate
            k = 10
            recall, ndcg = evaluate_rec_system(model, data, k=k)
            st.session_state.rec_metrics = {'recall': recall, 'ndcg': ndcg}
            
            progress_bar.empty()
            status_text.empty()
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train_losses)
            ax.set_title("Training Loss (BPR)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            
            results_container.pyplot(fig)
            plt.close()
            
            # Metrics
            cols = results_container.columns(3)
            cols[0].metric(f"Recall@{k}", f"{recall:.4f}")
            cols[1].metric(f"NDCG@{k}", f"{ndcg:.4f}")
            cols[2].metric("Time", f"{time.time()-start_time:.2f}s")


# Main app structure
import time # imported here again just in case, though it's at top imports

# Define tabs
tab1, tab2 = st.tabs(["📚 GNN Basics (Node Classification)", "🛍️ Recommender System (Link Prediction)"])

with tab1:
    show_gnn_basics_tab()

with tab2:
    show_recommendation_tab()
