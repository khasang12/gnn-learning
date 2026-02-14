from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_dataset(name="Cora", root="data"):
    """Load dataset"""
    dataset = Planetoid(root=f'{root}/{name}', name=name, transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data
