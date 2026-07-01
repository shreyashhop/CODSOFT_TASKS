"""Content-based filtering: recommend items similar to items a user liked,
based on item features (genres + description) rather than other users'
behavior. Good for new/unpopular items and explains "because you liked X".
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self._id_to_index = {}

    def fit(self, movies_df):
        """
        movies_df: DataFrame with columns [movie_id, title, genres, description]
        """
        self.movies_df = movies_df.reset_index(drop=True)
        self._id_to_index = {mid: idx for idx, mid in enumerate(self.movies_df["movie_id"])}

        # Combine genres (repeated for weight) and description into one text field.
        combined_text = (
            self.movies_df["genres"].str.replace("|", " ", regex=False) + " "
            + self.movies_df["genres"].str.replace("|", " ", regex=False) + " "  # upweight genres
            + self.movies_df["description"].fillna("")
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(combined_text)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return self

    def similar_items(self, movie_id, top_n=5):
        """Items most similar to a given item (e.g. for 'more like this')."""
        if movie_id not in self._id_to_index:
            raise ValueError(f"Unknown movie_id: {movie_id}")

        idx = self._id_to_index[movie_id]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        results = self.movies_df.iloc[[i for i, _ in scores]].copy()
        results["score"] = [s for _, s in scores]
        return results[["movie_id", "title", "genres", "score"]]

    def recommend_for_user(self, user_ratings, top_n=5, min_rating=4):
        """
        user_ratings: DataFrame/dict-like with columns [movie_id, rating] for one user.
        Builds a weighted "taste profile" from items the user rated highly, and
        recommends unseen items most similar to that profile.
        """
        if isinstance(user_ratings, pd.DataFrame):
            liked = user_ratings[user_ratings["rating"] >= min_rating]
        else:
            liked = pd.DataFrame(user_ratings)
            liked = liked[liked["rating"] >= min_rating]

        if liked.empty:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])

        seen_ids = set(user_ratings["movie_id"]) if isinstance(user_ratings, pd.DataFrame) \
            else {r["movie_id"] for r in user_ratings}

        # Weighted average of the TF-IDF vectors of liked items (weight = rating).
        liked_indices = [self._id_to_index[m] for m in liked["movie_id"] if m in self._id_to_index]
        weights = liked["rating"].values[: len(liked_indices)]

        if not liked_indices:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])

        profile = np.asarray(
            self.tfidf_matrix[liked_indices].multiply(weights.reshape(-1, 1)).mean(axis=0)
        )
        profile_similarity = cosine_similarity(profile, self.tfidf_matrix).flatten()

        scored = list(enumerate(profile_similarity))
        scored = [
            (i, s) for i, s in scored
            if self.movies_df.iloc[i]["movie_id"] not in seen_ids
        ]
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

        results = self.movies_df.iloc[[i for i, _ in scored]].copy()
        results["score"] = [s for _, s in scored]
        return results[["movie_id", "title", "genres", "score"]]
