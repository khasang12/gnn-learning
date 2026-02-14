#!/bin/bash
# Setup script for Streamlit Cloud deployment
# This script installs PyTorch Geometric and its dependencies

echo "Setting up environment for GNN Streamlit App..."

# Update pip
python -m pip install --upgrade pip

# Install PyTorch with CPU-only version (suitable for cloud)
# Streamlit Cloud uses Linux x86_64, so we install CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other requirements
pip install -r requirements.txt

echo "Setup completed successfully!"