import torch
import numpy as np

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
