"""End-to-end demo: loads sample data, trains every recommender, and prints
recommendations for a sample user.

Run: python main.py --user_id 1
"""

import argparse
import pandas as pd

from data_loader import load_movies, load_ratings
from content_based import ContentBasedRecommender
from collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization
from hybrid import HybridRecommender


def print_table(title, df, score_col):
    print(f"\n--- {title} ---")
    if df.empty:
        print("(no recommendations)")
        return
    for _, row in df.iterrows():
        print(f"  {row['title']:24s} [{row['genres']}]   score={row[score_col]:.2f}")


def main(user_id, top_n):
    movies_df = load_movies()
    ratings_df = load_ratings()
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]

    print(f"User {user_id}'s rated movies:")
    rated = user_ratings.merge(movies_df, on="movie_id")[["title", "genres", "rating"]]
    print(rated.sort_values("rating", ascending=False).to_string(index=False))

    # --- Content-based ---
    content_model = ContentBasedRecommender().fit(movies_df)
    content_recs = content_model.recommend_for_user(user_ratings, top_n=top_n)
    print_table("Content-Based Recommendations", content_recs, "score")

    # --- Collaborative filtering ---
    user_cf = UserBasedCF(k_neighbors=5).fit(ratings_df)
    user_cf_recs = user_cf.recommend_for_user(user_id, movies_df, top_n=top_n)
    print_table("User-Based CF Recommendations", user_cf_recs, "predicted_rating")

    item_cf = ItemBasedCF(k_neighbors=5).fit(ratings_df)
    item_cf_recs = item_cf.recommend_for_user(user_id, movies_df, top_n=top_n)
    print_table("Item-Based CF Recommendations", item_cf_recs, "predicted_rating")

    mf = MatrixFactorization(n_factors=8, n_epochs=40).fit(ratings_df)
    mf_recs = mf.recommend_for_user(user_id, ratings_df, movies_df, top_n=top_n)
    print_table("Matrix Factorization Recommendations", mf_recs, "predicted_rating")

    # --- Hybrid ---
    hybrid_model = HybridRecommender(content_model, user_cf, alpha=0.6)
    hybrid_recs = hybrid_model.recommend_for_user(user_id, ratings_df, movies_df, top_n=top_n)
    print_table("Hybrid Recommendations", hybrid_recs, "hybrid_score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo the recommendation system")
    parser.add_argument("--user_id", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    main(args.user_id, args.top_n)
