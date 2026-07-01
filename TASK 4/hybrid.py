"""Hybrid recommender: blends content-based and collaborative-filtering
scores. Useful because CF struggles on new users/items ("cold start") while
content-based has no such issue but can be less personalized; combining both
tends to outperform either alone.
"""

import pandas as pd


class HybridRecommender:
    def __init__(self, content_model, cf_model, alpha=0.5):
        """
        alpha: weight on the collaborative-filtering score (0..1).
               (1 - alpha) is the weight on the content-based score.
        """
        self.content_model = content_model
        self.cf_model = cf_model
        self.alpha = alpha

    def recommend_for_user(self, user_id, ratings_df, movies_df, top_n=5):
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        seen_ids = set(user_ratings["movie_id"])

        content_scores = self.content_model.recommend_for_user(
            user_ratings, top_n=len(movies_df)
        ).set_index("movie_id")["score"] if not user_ratings.empty else pd.Series(dtype=float)

        cf_scores_df = self.cf_model.recommend_for_user(user_id, movies_df, top_n=len(movies_df))
        cf_scores = cf_scores_df.set_index("movie_id")["predicted_rating"] if not cf_scores_df.empty else pd.Series(dtype=float)

        # Normalize each score column to 0-1 so they're comparable before blending.
        def normalize(s):
            if s.empty or s.max() == s.min():
                return s * 0
            return (s - s.min()) / (s.max() - s.min())

        content_norm = normalize(content_scores)
        cf_norm = normalize(cf_scores)

        all_ids = (set(content_norm.index) | set(cf_norm.index)) - seen_ids
        rows = []
        for mid in all_ids:
            c_score = content_norm.get(mid, 0.0)
            cf_score = cf_norm.get(mid, 0.0)
            blended = self.alpha * cf_score + (1 - self.alpha) * c_score
            rows.append((mid, blended))

        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
        result = movies_df[movies_df["movie_id"].isin([m for m, _ in rows])].copy()
        score_map = dict(rows)
        result["hybrid_score"] = result["movie_id"].map(score_map)
        return result.sort_values("hybrid_score", ascending=False)[
            ["movie_id", "title", "genres", "hybrid_score"]
        ]
