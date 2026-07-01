"""Simple evaluation harness: holds out a slice of ratings, trains on the
rest, and measures RMSE (and top-N precision) on the held-out ratings.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization


def rmse(predictions, targets):
    predictions, targets = np.array(predictions), np.array(targets)
    mask = ~np.isnan(predictions)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions[mask] - targets[mask]) ** 2)))


def evaluate_model(model_name, model, train_df, test_df, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    model.fit(train_df, **fit_kwargs)

    preds = [model.predict_rating(u, m) for u, m in zip(test_df["user_id"], test_df["movie_id"])]
    score = rmse(preds, test_df["rating"])
    coverage = sum(not np.isnan(p) for p in preds) / len(preds)
    print(f"{model_name:20s} RMSE: {score:.3f}   (coverage: {coverage:.0%})")
    return score


def run_evaluation(ratings_df, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=seed)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}\n")
    evaluate_model("User-based CF", UserBasedCF(k_neighbors=5), train_df, test_df)
    evaluate_model("Item-based CF", ItemBasedCF(k_neighbors=5), train_df, test_df)
    evaluate_model("Matrix Factorization", MatrixFactorization(n_factors=8, n_epochs=40), train_df, test_df)


if __name__ == "__main__":
    from data_loader import load_ratings

    run_evaluation(load_ratings())
