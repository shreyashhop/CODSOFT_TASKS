# Movie Recommendation System

A simple recommendation system demonstrating three classic approaches —
**content-based filtering**, **collaborative filtering** (user-based,
item-based, and matrix factorization), and a **hybrid** blend of the two —
applied to a small synthetic movie catalog. Swap in a real dataset
(MovieLens, your own product catalog, a book dataset, etc.) and it works the
same way.

## How it works

**1. Content-based filtering (`content_based.py`)**
Recommends items *similar to what a user already liked*, using the items'
own features (genres + description) rather than other users' behavior.
Genres/descriptions are vectorized with **TF-IDF**, and cosine similarity
finds the closest matches. A user's "taste profile" is the rating-weighted
average of the TF-IDF vectors of everything they rated highly. Works well
for new items with no ratings yet ("cold start" for items) and is easy to
explain ("recommended because you liked X").

**2. Collaborative filtering (`collaborative_filtering.py`)**
Recommends items based on **patterns across many users' ratings**, with no
knowledge of what the items actually are:
- **User-based CF** — finds users with similar rating patterns ("users like
  you"), and recommends what those neighbors rated highly.
- **Item-based CF** — finds items with similar rating patterns across users
  ("people who liked this also liked..."), generally more stable than
  user-based CF as the catalog grows.
- **Matrix Factorization** — learns low-dimensional latent "taste" vectors
  for users and "profile" vectors for items via stochastic gradient descent,
  so that `user_vector · item_vector ≈ rating`. This is a simplified version
  of the technique behind the Netflix Prize-winning models, and typically
  generalizes better than the neighbor-based methods above, especially on
  sparse data.

**3. Hybrid (`hybrid.py`)**
Blends normalized content-based and collaborative scores (weighted by
`alpha`). Combines CF's personalization with content-based's ability to
recommend items that have few or no ratings yet.

## Project structure

```
recommendation_system/
├── data_loader.py              # Generates/loads sample movies + ratings
├── content_based.py             # TF-IDF + cosine similarity recommender
├── collaborative_filtering.py    # User-based CF, item-based CF, matrix factorization
├── hybrid.py                      # Blends content-based + CF scores
├── evaluate.py                     # Train/test split + RMSE evaluation
├── main.py                          # End-to-end demo script
├── requirements.txt
└── data/                              # Generated CSVs land here
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python main.py --user_id 1 --top_n 5
```

This prints the user's existing ratings, then recommendations from every
method side by side, e.g.:

```
--- Content-Based Recommendations ---
  Second Chances           [Romance|Comedy]   score=0.27
  Summit                   [Adventure|Drama]   score=0.16
  ...

--- Matrix Factorization Recommendations ---
  Neon Circuit             [Sci-Fi|Thriller]   score=3.58
  ...

--- Hybrid Recommendations ---
  Wildwood                 [Fantasy|Drama|Adventure]   score=0.80
  ...
```

## Evaluate accuracy

```bash
python evaluate.py
```

Holds out 20% of ratings, trains each collaborative model on the rest, and
reports **RMSE** (lower is better) plus prediction coverage:

```
User-based CF        RMSE: 0.854   (coverage: 100%)
Item-based CF        RMSE: 0.835   (coverage: 100%)
Matrix Factorization  RMSE: 0.814   (coverage: 100%)
```

## Using your own data

Replace the synthetic catalog in `data_loader.py` with real data by loading
CSVs shaped like:

```
movies.csv:  movie_id, title, genres, description
ratings.csv: user_id, movie_id, rating   (1-5 scale)
```

For books or general products, just rename the columns conceptually
(`genres` → `categories/tags`, `description` → product description) — the
same TF-IDF and collaborative-filtering logic applies unchanged.

## Design notes / extension ideas

- **Cold start:** brand-new users (no ratings yet) get nothing from CF —
  fall back to popularity-based or content-based recommendations until they
  rate a few items.
- **Implicit feedback:** if you only have clicks/purchases (no explicit
  1-5 ratings), treat any interaction as a positive signal and adapt the
  matrix factorization loss accordingly (e.g. weighted ALS).
- **Scaling up:** for large catalogs, exact cosine similarity over the full
  item-item or user-user matrix gets expensive — approximate nearest
  neighbor libraries (e.g. Annoy, FAISS) or `scipy.sparse` matrices help.
- **Richer content features:** add cast/crew, release year, or a
  learned embedding (e.g. sentence-transformers on the description) instead
  of pure TF-IDF for better similarity quality.
- **A/B testing in production:** offline RMSE doesn't always predict
  business impact (click-through, watch time) — validate with online
  experiments once a model looks good offline.
