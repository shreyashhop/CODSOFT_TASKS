"""Collaborative filtering: recommend items based on patterns across many
users' ratings, without needing any item content/features.

Includes three approaches:
  - UserBasedCF: "users like you also liked..."
  - ItemBasedCF: "people who liked this item also liked..."
  - MatrixFactorization: learns latent taste/item vectors via SGD (like a
    simplified version of the technique behind the Netflix Prize models).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings_df):
    """Pivot long-format ratings into a (users x items) matrix, 0 = unrated."""
    matrix = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
    return matrix


class UserBasedCF:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        self.matrix = None
        self.user_similarity = None

    def fit(self, ratings_df):
        self.matrix = build_user_item_matrix(ratings_df)
        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.matrix.values),
            index=self.matrix.index, columns=self.matrix.index,
        )
        return self

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.matrix.index or movie_id not in self.matrix.columns:
            return np.nan

        sims = self.user_similarity[user_id].drop(user_id)
        raters = self.matrix[movie_id]
        raters = raters[raters > 0]
        sims = sims[raters.index]

        top_sims = sims.sort_values(ascending=False).head(self.k_neighbors)
        top_sims = top_sims[top_sims > 0]
        if top_sims.empty:
            return np.nan

        ratings = raters.loc[top_sims.index]
        return float(np.dot(top_sims, ratings) / top_sims.sum())

    def recommend_for_user(self, user_id, movies_df, top_n=5):
        if user_id not in self.matrix.index:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "predicted_rating"])

        unseen = self.matrix.columns[self.matrix.loc[user_id] == 0]
        preds = [(mid, self.predict_rating(user_id, mid)) for mid in unseen]
        preds = [(mid, p) for mid, p in preds if not np.isnan(p)]
        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

        result = movies_df[movies_df["movie_id"].isin([m for m, _ in preds])].copy()
        score_map = dict(preds)
        result["predicted_rating"] = result["movie_id"].map(score_map)
        return result.sort_values("predicted_rating", ascending=False)[
            ["movie_id", "title", "genres", "predicted_rating"]
        ]


class ItemBasedCF:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        self.matrix = None
        self.item_similarity = None

    def fit(self, ratings_df):
        self.matrix = build_user_item_matrix(ratings_df)
        self.item_similarity = pd.DataFrame(
            cosine_similarity(self.matrix.values.T),
            index=self.matrix.columns, columns=self.matrix.columns,
        )
        return self

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.matrix.index or movie_id not in self.matrix.columns:
            return np.nan

        user_ratings = self.matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        if rated_items.empty:
            return np.nan

        sims = self.item_similarity[movie_id][rated_items.index]
        top_sims = sims.sort_values(ascending=False).head(self.k_neighbors)
        top_sims = top_sims[top_sims > 0]
        if top_sims.empty:
            return np.nan

        ratings = rated_items.loc[top_sims.index]
        return float(np.dot(top_sims, ratings) / top_sims.sum())

    def recommend_for_user(self, user_id, movies_df, top_n=5):
        if user_id not in self.matrix.index:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "predicted_rating"])

        unseen = self.matrix.columns[self.matrix.loc[user_id] == 0]
        preds = [(mid, self.predict_rating(user_id, mid)) for mid in unseen]
        preds = [(mid, p) for mid, p in preds if not np.isnan(p)]
        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

        result = movies_df[movies_df["movie_id"].isin([m for m, _ in preds])].copy()
        score_map = dict(preds)
        result["predicted_rating"] = result["movie_id"].map(score_map)
        return result.sort_values("predicted_rating", ascending=False)[
            ["movie_id", "title", "genres", "predicted_rating"]
        ]


class MatrixFactorization:
    """Learns low-rank user and item latent-factor matrices via SGD so that
    user_vec . item_vec ~= rating, filling in the sparse ratings matrix.
    """

    def __init__(self, n_factors=10, lr=0.01, reg=0.02, n_epochs=50, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)

        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0
        self._user_to_idx = {}
        self._item_to_idx = {}
        self._idx_to_item = {}

    def fit(self, ratings_df):
        users = ratings_df["user_id"].unique()
        items = ratings_df["movie_id"].unique()
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {m: i for i, m in enumerate(items)}
        self._idx_to_item = {i: m for m, i in self._item_to_idx.items()}

        n_users, n_items = len(users), len(items)
        self.user_factors = self.rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = self.rng.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = ratings_df["rating"].mean()

        samples = [
            (self._user_to_idx[u], self._item_to_idx[m], r)
            for u, m, r in ratings_df[["user_id", "movie_id", "rating"]].itertuples(index=False)
        ]

        for _ in range(self.n_epochs):
            self.rng.shuffle(samples)
            for u_idx, i_idx, rating in samples:
                pred = (
                    self.global_mean + self.user_bias[u_idx] + self.item_bias[i_idx]
                    + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
                )
                error = rating - pred

                self.user_bias[u_idx] += self.lr * (error - self.reg * self.user_bias[u_idx])
                self.item_bias[i_idx] += self.lr * (error - self.reg * self.item_bias[i_idx])

                u_f = self.user_factors[u_idx].copy()
                self.user_factors[u_idx] += self.lr * (error * self.item_factors[i_idx] - self.reg * u_f)
                self.item_factors[i_idx] += self.lr * (error * u_f - self.reg * self.item_factors[i_idx])

        return self

    def predict_rating(self, user_id, movie_id):
        if user_id not in self._user_to_idx or movie_id not in self._item_to_idx:
            return np.nan
        u_idx, i_idx = self._user_to_idx[user_id], self._item_to_idx[movie_id]
        pred = (
            self.global_mean + self.user_bias[u_idx] + self.item_bias[i_idx]
            + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        )
        return float(np.clip(pred, 1, 5))

    def recommend_for_user(self, user_id, ratings_df, movies_df, top_n=5):
        if user_id not in self._user_to_idx:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "predicted_rating"])

        seen = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"])
        unseen_items = [m for m in self._item_to_idx if m not in seen]

        preds = [(m, self.predict_rating(user_id, m)) for m in unseen_items]
        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

        result = movies_df[movies_df["movie_id"].isin([m for m, _ in preds])].copy()
        score_map = dict(preds)
        result["predicted_rating"] = result["movie_id"].map(score_map)
        return result.sort_values("predicted_rating", ascending=False)[
            ["movie_id", "title", "genres", "predicted_rating"]
        ]
