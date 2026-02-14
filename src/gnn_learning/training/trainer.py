import torch
import numpy as np
import time
from ..models.lightgcn import bpr_loss

def train_model(model, data, optimizer, criterion, num_epochs, progress_callback=None):
    """Train the model with progress tracking
    progress_callback: function that accepts (progress, status_text)
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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
        if progress_callback:
            progress = epoch / num_epochs
            status_text = f"Epoch {epoch}/{num_epochs} - Train Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}"
            progress_callback(progress, status_text)

    return train_losses, val_losses, train_accs, val_accs


def train_lightgcn(model, data, optimizer, num_epochs=100, batch_size=1024, progress_callback=None):
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
            
        # Update progress
        if progress_callback:
            progress = epoch / num_epochs
            status_text = f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}"
            progress_callback(progress, status_text)

    return train_losses
