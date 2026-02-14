#!/usr/bin/env python3
"""
Complete GNN Example for Intermediate AI Learners

This example demonstrates:
1. Loading a graph dataset (Cora)
2. Implementing a Graph Convolutional Network (GCN)
3. Training and evaluating the model
4. Visualizing results

Requirements: PyTorch, PyTorch Geometric, matplotlib, scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_cora_dataset():
    """
    Load the Cora citation network dataset.
    
    Returns:
        dataset: PyG Dataset object
        data: PyG Data object with node features, edge indices, and labels
    """
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f'Dataset: {dataset}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of node features: {data.num_node_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Training nodes: {data.train_mask.sum().item()}')
    print(f'Validation nodes: {data.val_mask.sum().item()}')
    print(f'Test nodes: {data.test_mask.sum().item()}')
    
    return dataset, data

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for node classification.
    
    Architecture:
    - Two GCNConv layers with ReLU activation and dropout
    - Final GCNConv layer for classification
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer with ReLU activation and dropout
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer (output layer)
        x = self.gcn2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, criterion):
    """
    Train the model for one epoch.
    
    Args:
        model: GNN model
        data: PyG Data object
        optimizer: PyTorch optimizer
        criterion: loss function
    
    Returns:
        loss: training loss
        accuracy: training accuracy
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)
    
    # Compute loss only on training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    pred = out[data.train_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    accuracy = correct / data.train_mask.sum().item()
    
    return loss.item(), accuracy

def evaluate(model, data, mask):
    """
    Evaluate the model on a given mask (validation or test).
    
    Args:
        model: GNN model
        data: PyG Data object
        mask: boolean mask indicating which nodes to evaluate
    
    Returns:
        loss: evaluation loss
        accuracy: evaluation accuracy
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = F.nll_loss(out[mask], data.y[mask]).item()
        pred = out[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()
    return loss, accuracy

def visualize_embeddings(model, data, epoch=None):
    """
    Visualize node embeddings using t-SNE.
    
    Args:
        model: GNN model
        data: PyG Data object
        epoch: current epoch number (for title)
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings after first GCN layer
        x, edge_index = data.x, data.edge_index
        embeddings = model.gcn1(x, edge_index).cpu().numpy()
        labels = data.y.cpu().numpy()
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab20', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Class')
        title = f't-SNE visualization of node embeddings'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        plt.title(title)
        plt.xlabel('t-SNE component 1')
        plt.ylabel('t-SNE component 2')
        plt.tight_layout()
        plt.savefig(f'embeddings_epoch_{epoch if epoch else "final"}.png')
        plt.close()

def main():
    """Main training and evaluation loop."""
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    dataset, data = load_cora_dataset()
    data = data.to(device)
    
    # Create model
    num_features = dataset.num_node_features
    hidden_dim = 16
    num_classes = dataset.num_classes
    
    model = GCN(num_features, hidden_dim, num_classes).to(device)
    print(f'Model architecture:\n{model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    # Training loop
    num_epochs = 200
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f'\nStarting training for {num_epochs} epochs...')
    for epoch in tqdm(range(1, num_epochs + 1)):
        # Train
        train_loss, train_acc = train(model, data, optimizer, criterion)
        
        # Validate
        val_loss, val_acc = evaluate(model, data, data.val_mask)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Visualize embeddings every 50 epochs
        if epoch % 50 == 0:
            visualize_embeddings(model, data, epoch)
    
    # Final test evaluation
    test_loss, test_acc = evaluate(model, data, data.test_mask)
    print(f'\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Visualize final embeddings
    visualize_embeddings(model, data, 'final')
    
    # Save model
    torch.save(model.state_dict(), 'gcn_cora_model.pth')
    print(f'Model saved to gcn_cora_model.pth')
    
    # Print summary
    print('\n' + '='*50)
    print('SUMMARY:')
    print('='*50)
    print(f'Dataset: Cora citation network')
    print(f'Model: GCN with 2 layers (hidden_dim={hidden_dim})')
    print(f'Training epochs: {num_epochs}')
    print(f'Final Test Accuracy: {test_acc:.4f}')
    print(f'Visualizations saved:')
    print(f'  - training_history.png (loss/accuracy curves)')
    print(f'  - embeddings_epoch_*.png (t-SNE visualizations)')
    print(f'  - gcn_cora_model.pth (trained model weights)')
    print('='*50)

if __name__ == '__main__':
    main()