"""Loads (or generates) a small movies + ratings dataset to demo the
recommendation system without needing an external download.

Swap `load_movies()` / `load_ratings()` for real data (e.g. MovieLens) by
pointing them at CSVs with the same columns:
    movies.csv:  movie_id, title, genres, description
    ratings.csv: user_id, movie_id, rating   (rating: 1-5)
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

# A small, hand-crafted catalog spanning a few genres, so content-based
# similarity has something meaningful to work with.
_MOVIES = [
    (1, "Galaxy Raiders", "Sci-Fi|Action|Adventure", "A ragtag crew battles an empire across the stars."),
    (2, "The Last Starship", "Sci-Fi|Drama", "A lone pilot searches for humanity's new home."),
    (3, "Neon Circuit", "Sci-Fi|Thriller", "A hacker uncovers a conspiracy inside a mega-corporation."),
    (4, "Love in Paris", "Romance|Drama", "Two strangers fall in love over a rainy Parisian week."),
    (5, "Second Chances", "Romance|Comedy", "Former sweethearts reunite at a wedding gone wrong."),
    (6, "Heart of the City", "Romance|Drama", "A chef and a violinist find love amid busy city life."),
    (7, "Midnight Heist", "Action|Crime|Thriller", "A crew plans the perfect casino robbery."),
    (8, "Steel Vengeance", "Action|Crime", "An ex-cop hunts down the gang that framed him."),
    (9, "Silent Witness", "Thriller|Mystery", "A journalist unravels a decades-old murder case."),
    (10, "The Quiet Detective", "Mystery|Drama", "A retired detective is pulled into one last case."),
    (11, "Laugh Track", "Comedy", "A failing sitcom writer gets one last shot at a hit show."),
    (12, "Roommates", "Comedy|Romance", "Two mismatched roommates slowly become inseparable."),
    (13, "Haunted Hollow", "Horror|Thriller", "A family moves into a house with a dark history."),
    (14, "The Last Signal", "Horror|Sci-Fi", "A research station picks up a signal that shouldn't exist."),
    (15, "Kingdom of Ash", "Fantasy|Adventure", "A young heir must reclaim a kingdom from a usurper."),
    (16, "The Dragon's Oath", "Fantasy|Action", "A knight and a dragon form an uneasy alliance."),
    (17, "Wildwood", "Fantasy|Drama|Adventure", "A girl discovers a hidden forest kingdom near her home."),
    (18, "Deep Blue", "Adventure|Drama", "A marine biologist races to save a dying coral reef."),
    (19, "Summit", "Adventure|Drama", "Climbers attempt an unclimbed Himalayan peak."),
    (20, "Startup City", "Comedy|Drama", "A group of friends chase a tech dream that keeps slipping away."),
]

_GENRES = sorted({g for _, _, genres, _ in _MOVIES for g in genres.split("|")})


def _generate_ratings(num_users=15, seed=42):
    """Synthesize ratings: each user has genre preferences plus some noise,
    so both content-based and collaborative signals are present."""
    rng = np.random.default_rng(seed)
    movies_df = load_movies()

    rows = []
    for user_id in range(1, num_users + 1):
        # Each user likes 1-3 genres more than others.
        liked_genres = rng.choice(_GENRES, size=rng.integers(1, 3), replace=False)
        n_rated = rng.integers(6, 12)
        rated_movies = rng.choice(movies_df["movie_id"], size=n_rated, replace=False)

        for movie_id in rated_movies:
            movie = movies_df[movies_df["movie_id"] == movie_id].iloc[0]
            movie_genres = set(movie["genres"].split("|"))
            overlap = len(movie_genres & set(liked_genres))

            base = 2.5 + overlap * 1.0  # more overlap with liked genres -> higher rating
            noise = rng.normal(0, 0.6)
            rating = int(np.clip(round(base + noise), 1, 5))
            rows.append((user_id, int(movie_id), rating))

    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def load_movies():
    if os.path.exists(MOVIES_CSV):
        return pd.read_csv(MOVIES_CSV)
    return pd.DataFrame(_MOVIES, columns=["movie_id", "title", "genres", "description"])


def load_ratings():
    if os.path.exists(RATINGS_CSV):
        return pd.read_csv(RATINGS_CSV)
    return _generate_ratings()


def save_sample_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    movies_df = load_movies()
    ratings_df = load_ratings()
    movies_df.to_csv(MOVIES_CSV, index=False)
    ratings_df.to_csv(RATINGS_CSV, index=False)
    return movies_df, ratings_df


if __name__ == "__main__":
    movies_df, ratings_df = save_sample_data()
    print(f"Saved {len(movies_df)} movies to {MOVIES_CSV}")
    print(f"Saved {len(ratings_df)} ratings to {RATINGS_CSV}")
