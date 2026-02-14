# Complete GNN Example for Intermediate AI Learners

This repository provides a complete, working example of a Graph Neural Network (GNN) implemented in Python using PyTorch and PyTorch Geometric. It's designed for intermediate AI learners who want to understand how GNNs work in practice.

## Features

- **Complete Implementation**: End-to-end GNN example from data loading to visualization
- **Educational Focus**: Well-documented code with explanations for each component
- **Interactive Web Interface**: Streamlit app for experimenting with hyperparameters
- **Visualization**: t-SNE embeddings and training history plots
- **Reproducible**: Random seeds set for consistent results
- **Real Dataset**: Uses the Cora citation network dataset

## What You'll Learn

1. How to load and preprocess graph data using PyTorch Geometric
2. How to implement a Graph Convolutional Network (GCN) for node classification
3. How to train and evaluate a GNN model
4. How to visualize node embeddings and training progress
5. How to handle graph-structured data in machine learning

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- matplotlib
- scikit-learn
- numpy
- tqdm
- streamlit>=1.28.0
- pillow>=9.0.0

## Quick Start

### 1. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Example

```bash
python gnn_example.py
```

The script will:
- Download the Cora dataset (automatically)
- Train a GCN model for 200 epochs
- Evaluate on test set
- Generate visualizations
- Save the trained model

## Interactive Web Interface (Streamlit)

For an interactive learning experience, we provide a Streamlit web application:

### 1. Make sure dependencies are installed (includes streamlit)

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
streamlit run gnn_streamlit.py
```

The app will open in your browser and provides:
- **Interactive Controls**: Adjust hyperparameters (hidden dimension, dropout, learning rate, etc.)
- **Real-time Training Visualization**: Watch loss and accuracy curves update as training progresses
- **Dataset Exploration**: View dataset statistics and t-SNE embeddings
- **No Coding Required**: Perfect for experimenting with different configurations

## Project Structure

```
.
├── gnn_example.py          # Main GNN implementation (command line)
├── gnn_streamlit.py       # Interactive web interface (Streamlit)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── test_gnn.py            # Basic tests
├── simple_test.py         # Environment verification
├── verify_gnn.py          # Full verification with synthetic data
├── verification_training_history.png  # Sample output
└── venv/                  # Virtual environment (created)
```

## Code Overview

### Key Components

1. **Data Loading**: Uses PyTorch Geometric's `Planetoid` dataset to load Cora
2. **GCN Model**: Two-layer Graph Convolutional Network with dropout
3. **Training Loop**: Standard PyTorch training with validation monitoring
4. **Evaluation**: Separate test evaluation with accuracy reporting
5. **Visualization**: t-SNE plots of node embeddings at different training stages

### Model Architecture

```python
GCN(
  (gcn1): GCNConv(1433, 16)
  (gcn2): GCNConv(16, 7)
)
```

- Input: 1,433-dimensional node features
- Hidden layer: 16-dimensional embeddings
- Output: 7 classes (paper topics in Cora)

## Expected Output

When you run the script, you should see output like:

```
Loading Cora dataset...
Dataset: Cora()
Number of graphs: 1
Number of nodes: 2708
Number of edges: 10556
Number of node features: 1433
Number of classes: 7
Training nodes: 140
Validation nodes: 500
Test nodes: 1000

Starting training for 200 epochs...
Epoch 020: Train Loss: 1.8765, Train Acc: 0.1929, Val Loss: 1.8643, Val Acc: 0.3040
Epoch 040: Train Loss: 1.7304, Train Acc: 0.4000, Val Loss: 1.7510, Val Acc: 0.4600
...
Epoch 200: Train Loss: 0.2405, Train Acc: 0.9857, Val Loss: 1.1273, Val Acc: 0.7980

Final Test Results:
Test Loss: 1.0987, Test Accuracy: 0.8050
```

## Generated Files

After running the script, you'll get:

1. **training_history.png** - Loss and accuracy curves over epochs
2. **embeddings_epoch_*.png** - t-SNE visualizations at epochs 50, 100, 150, 200
3. **gcn_cora_model.pth** - Trained model weights for reuse
4. **data/Cora/** - Downloaded dataset (created automatically)

## Understanding the Results

- **Cora Dataset**: Citation network where nodes are papers, edges are citations
- **Task**: Predict the topic category of each paper (7 classes)
- **Typical Performance**: ~80-85% test accuracy with this simple GCN
- **Visualizations**: Show how embeddings become more separable as training progresses

## Extending the Example

You can modify the code to:

1. **Try different GNN architectures**: GAT, GraphSAGE, etc.
2. **Use different datasets**: CiteSeer, PubMed, or custom graphs
3. **Add more layers**: Experiment with depth and dropout
4. **Implement early stopping**: Stop training when validation loss plateaus
5. **Add hyperparameter tuning**: Use Optuna or Grid Search

## Troubleshooting

### PyTorch Geometric Installation Issues

If you encounter issues installing PyTorch Geometric, you may need to install it from source:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

Replace `cu118` with your CUDA version if using GPU.

### Dataset Download Issues

If the dataset fails to download, check your internet connection. You can also manually download from:
https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

## Deploy on Streamlit Cloud

You can deploy this interactive GNN learning app on Streamlit Community Cloud for free:

### 1. Prepare Your Repository

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit: GNN Learning App"

# Create a GitHub repository and push your code
git remote add origin https://github.com/your-username/gnn-learning-app.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Streamlit Cloud

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Click "New app" and connect your GitHub repository
3. Select the repository and branch
4. Set the main file path to `gnn_streamlit.py`
5. Click "Deploy"

### 3. Streamlit Cloud Configuration Files

The repository includes:
- `requirements.txt` - Python dependencies
- `setup.sh` - Optional setup script for PyTorch Geometric
- `.streamlit/config.toml` - Streamlit configuration
- `.gitignore` - Git ignore file

### 4. Troubleshooting Deployment

If you encounter issues on Streamlit Cloud:

1. **PyTorch Geometric installation**: The `setup.sh` script handles installation of PyTorch Geometric dependencies
2. **Memory limits**: Streamlit Cloud has memory limits; the app is optimized to work within these constraints
3. **Dataset caching**: The app caches the dataset after first download for faster subsequent loads

## Learning Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [GNN Papers with Code](https://paperswithcode.com/task/graph-neural-networks)
- [Stanford CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Cloud](https://streamlit.io/cloud)

## License

This educational example is released under the MIT License.

## Contributing

Feel free to submit issues or pull requests to improve this educational example!

## Project Status

✅ **Local Development**: Fully functional  
✅ **Interactive Web App**: Streamlit interface available  
✅ **Deployment Ready**: Configured for Streamlit Cloud  
🚀 **Ready for Production**: Deploy with one click to share with others!
