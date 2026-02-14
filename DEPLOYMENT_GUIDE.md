# Streamlit Cloud Deployment Guide

This guide walks you through deploying the GNN Learning App on Streamlit Community Cloud.

## Prerequisites

1. **GitHub Account**: You need a GitHub account to host your code
2. **Streamlit Account**: Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)

## Step-by-Step Deployment

### 1. Prepare Your Local Repository

```bash
# Navigate to the project directory
cd /Users/sangkha/Documents/Study/Master/HK252/GNN\ Learning

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit: GNN Learning App with Streamlit interface"

# Check your files are staged
git status
```

### 2. Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository (e.g., `gnn-learning-app`)
4. Choose public or private (Streamlit Cloud supports both)
5. **DO NOT** initialize with README, .gitignore, or license (you already have these)
6. Click "Create repository"

### 3. Connect and Push to GitHub

```bash
# Add the remote repository (replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/gnn-learning-app.git

# Rename branch to main
git branch -M main

# Push your code
git push -u origin main
```

### 4. Deploy on Streamlit Cloud

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/gnn-learning-app`
5. Select branch: `main`
6. Set main file path: `gnn_streamlit.py`
7. Click "Deploy"

## Deployment Configuration

### Required Files for Streamlit Cloud

Your repository contains these essential files:

1. **`gnn_streamlit.py`** - Main Streamlit application with tabs for GNN basics and recommendations
2. **`requirements.txt`** - Python dependencies
3. **`.streamlit/config.toml`** - Streamlit configuration
4. **`.gitignore`** - Files to exclude from Git
5. **`setup.sh`** - Optional setup script for PyTorch Geometric

### How Streamlit Cloud Works

1. **Automatic Environment Setup**: Streamlit Cloud reads `requirements.txt` and installs all dependencies
2. **App Launch**: Runs `streamlit run gnn_streamlit.py` automatically
3. **Public URL**: Your app gets a public URL like `https://YOUR_APP.streamlit.app`

## Troubleshooting Deployment Issues

### Common Issues and Solutions

#### 1. PyTorch Geometric Installation Failures
**Problem**: PyTorch Geometric has complex dependencies
**Solution**: The `setup.sh` script handles this. Streamlit Cloud will run it automatically.

#### 2. Memory Limits
**Problem**: Streamlit Cloud has memory limits (1GB for free tier)
**Solution**: The app is optimized:
- Uses CPU-only PyTorch
- Caches dataset after first download
- Limits training to reasonable epochs

#### 3. Long Deployment Time
**Problem**: PyTorch installation can take 5-10 minutes
**Solution**: This is normal. Streamlit Cloud caches dependencies after first deployment.

#### 4. Dataset Download Issues
**Problem**: Cora dataset fails to download
**Solution**: The app includes retry logic and caching. First load may be slow.

## Testing Before Deployment

### Local Testing Commands

```bash
# Test basic functionality
python test_streamlit_import.py

# Test GNN core functionality
python verify_gnn.py

# Run Streamlit app locally
streamlit run gnn_streamlit.py
```

### Expected Local Output

When running locally, you should see:
```
$ streamlit run gnn_streamlit.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

## Post-Deployment

### Monitoring Your App

1. **Streamlit Cloud Dashboard**: View app usage and logs
2. **App Settings**: Configure secrets, environment variables
3. **Version Management**: Roll back to previous versions if needed

### Updating Your App

```bash
# Make changes to your code
git add .
git commit -m "Update app with new features"
git push origin main

# Streamlit Cloud automatically redeploys
```

## Advanced Configuration

### Environment Variables

Add a `.streamlit/secrets.toml` file for sensitive data:

```toml
# .streamlit/secrets.toml
MY_SECRET_KEY = "your-secret-key-here"
```

### Custom Domain (Pro Feature)

Upgrade to Streamlit Pro to use custom domains.

## Support Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

## Success Checklist

- [ ] Git repository initialized and committed
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App deployed on Streamlit Cloud
- [ ] App loads successfully
- [ ] GNN training works
- [ ] Recommendation system demo works
- [ ] Visualizations display correctly

## Need Help?

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are in the repository
3. Ensure `requirements.txt` is correct
4. Contact Streamlit support via their community forum

Your GNN Learning App will be live at: `https://YOUR_APP.streamlit.app`

Happy deploying! 🚀