# GNN Learning Project

This project implements Graph Neural Networks (GNNs) using PyTorch Geometric, structured as a modular Python package with a Streamlit interface.

## Project Structure

The project has been restructured for better maintainability and scalability:

```
.
├── src/
│   └── gnn_learning/       # Core package
│       ├── models/         # GNN models (GCN, LightGCN)
│       ├── data/           # Data loading and generation
│       ├── training/       # Training loops
│       ├── evaluation/     # Evaluation metrics
│       └── visualization/  # Plotting and unique visuals
├── app/
│   └── streamlit_app.py    # Streamlit web application
├── tests/                  # Unit tests
├── pyproject.toml          # Project configuration and dependencies
└── uv.lock                 # Dependency lock file
```

## Features

- **Modular Architecture**: Separate modules for models, data, training, and evaluation.
- **Dependency Management**: Uses `uv` for fast and reliable package management.
- **Interactive UI**: Streamlit app for GNode classification and Recommendation systems.
- **Testing**: Comprehensive test suite using `pytest`.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (recommended) or Python 3.10+

### Installation

1.  **Initialize the environment and install dependencies**:

    ```bash
    uv sync
    ```

    This will create a virtual environment and install all required packages.

### Running the App

To launch the interactive Streamlit application:

```bash
uv run streamlit run app/streamlit_app.py
```

### Running Tests

To run the test suite:

```bash
uv run pytest tests/
```

## Development

- **Adding dependencies**: `uv add <package>`
- **Running scripts**: `uv run <script>`

## License

MIT
