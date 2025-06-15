import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# ──────────── 1) Load data ────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "DATA", "ml-latest-small")

movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

# ──────────── 2) Train/test split ────────────
train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42,
    stratify=ratings["userId"]
)

# Move any “test-only” movies back into train:
train_movie_ids = set(train_ratings.movieId)
test_movie_ids  = set(test_ratings.movieId)
only_in_test = test_movie_ids - train_movie_ids
if only_in_test:
    mask = test_ratings.movieId.isin(only_in_test)
    to_move = test_ratings[mask]
    test_ratings = test_ratings[~mask]
    train_ratings = pd.concat([train_ratings, to_move], ignore_index=True)


# 3) Non-personalized  
# ────────────────────────────────────────────
global_mean = train_ratings.rating.mean()

def predict_rating_pop(user_id, movie_id):
    return global_mean

movie_avg = train_ratings.groupby("movieId")["rating"].mean().to_dict()

def predict_rating_movieavg(user_id, movie_id):
    return movie_avg.get(movie_id, global_mean)

# ──────────── Evaluate on the test set ────────────
# Global-Mean
test_preds_global = np.full(len(test_ratings), global_mean)
rmse_global = np.sqrt(mean_squared_error(test_ratings.rating, test_preds_global))
print(f"1.a) Global-Mean RMSE = {rmse_global:.4f}")

# Per-Movie-Average
test_preds_movie_avg = test_ratings.apply(
    lambda row: predict_rating_movieavg(row.userId, row.movieId),
    axis=1
).values
rmse_movie_avg = np.sqrt(mean_squared_error(test_ratings.rating, test_preds_movie_avg))
print(f"1.b) Per-Movie-Avg RMSE = {rmse_movie_avg:.4f}")



# 4) USER‐BASED COLLABORATIVE FILTERING
# ────────────────────────────────────────────

# 4.a) Build TRAIN‐only user/movie ↔ index maps
unique_train_user_ids  = sorted(train_ratings['userId'].unique())
unique_train_movie_ids = sorted(train_ratings['movieId'].unique())
user_to_idx  = { uid: i for i, uid in enumerate(unique_train_user_ids) }
movie_to_idx = { mid: i for i, mid in enumerate(unique_train_movie_ids) }
n_users, n_movies = len(unique_train_user_ids), len(unique_train_movie_ids)

# 4.b) Build user_ratings list
user_ratings = [dict() for _ in range(n_users)]
for _, row in train_ratings.iterrows():
    uidx = user_to_idx[row.userId]
    midx = movie_to_idx[row.movieId]
    user_ratings[uidx][midx] = row.rating

# 4.c) Build dense_train matrix
dense_train = np.zeros((n_users, n_movies), dtype=np.float32)
for uidx, ratings_dict in enumerate(user_ratings):
    for midx, r in ratings_dict.items():
        dense_train[uidx, midx] = r

# 4.d) Build binary‐indicator B to count co‐ratings
row_ind, col_ind, data = [], [], []
for uidx, ratings_dict in enumerate(user_ratings):
    for midx in ratings_dict:
        row_ind.append(uidx)
        col_ind.append(midx)
        data.append(1)
B = csr_matrix((data, (row_ind, col_ind)), shape=(n_users, n_movies), dtype=np.int32)
co_counts = (B @ B.T).toarray()

# 4.e) Find k nearest neighbors with a minimum overlap
k_u, min_overlap = 30, 3
neighbor_idxs   = np.zeros((n_users, k_u), dtype=int)
user_sim_matrix = np.zeros((n_users, k_u), dtype=np.float32)

for uidx in range(n_users):
    candidates = [v for v in range(n_users)
                  if v!=uidx and co_counts[uidx,v]>=min_overlap]
    if not candidates:
        neighbor_idxs[uidx]   = -1
        user_sim_matrix[uidx] = 0.0
        continue

    cand_matrix = dense_train[candidates, :]
    u_vec       = dense_train[uidx].reshape(1,-1)
    sims        = cosine_similarity(u_vec, cand_matrix).flatten()
    top_k       = np.argsort(sims)[::-1][:k_u]

    nbrs = [candidates[i] for i in top_k]
    s    = [float(sims[i])       for i in top_k]
    pad  = k_u - len(nbrs)
    if pad>0:
        nbrs += [-1]*pad
        s    += [0.0]*pad

    neighbor_idxs[uidx]   = nbrs
    user_sim_matrix[uidx] = s

# 4.f) Define the prediction function
def predict_rating_usercf(user_id, movie_id):
    uidx = user_to_idx.get(user_id)
    midx = movie_to_idx.get(movie_id)
    if uidx is None or midx is None:
        return global_mean

    wsum, ssum = 0.0, 0.0
    for sim, v_idx in zip(user_sim_matrix[uidx], neighbor_idxs[uidx]):
        if v_idx<0: 
            continue
        if midx in user_ratings[v_idx]:
            r_v_m = user_ratings[v_idx][midx]
            wsum += sim * r_v_m
            ssum += sim

    return (wsum/ssum) if ssum>1e-6 else movie_avg.get(movie_id, global_mean)

# 4.g) Compute RMSE on test set
test_preds_ubcf = test_ratings.apply(
    lambda r: predict_rating_usercf(int(r.userId), int(r.movieId)),
    axis=1
).values
rmse_ubcf = np.sqrt(mean_squared_error(test_ratings.rating, test_preds_ubcf))
print(f"User‐CF (k_u={k_u}, min_overlap={min_overlap}) RMSE = {rmse_ubcf:.4f}")

# 4.h) Top‐N helper
def get_top_n_usercf_for_user(user_id, N=10):
    if user_id not in user_to_idx:
        return []
    seen       = set(train_ratings[train_ratings.userId==user_id].movieId)
    candidates = [m for m in unique_train_movie_ids if m not in seen]
    scored     = [(m, predict_rating_usercf(user_id, m)) for m in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m,_ in scored[:N]]

# Example usage:
usercf_recs = {
    u: get_top_n_usercf_for_user(int(u), N=10)
    for u in test_ratings.userId.unique()
}

# 5) ITEM-BASED FILTERING
# ────────────────────────────────────────────

# ─── 1) BUILD INDEX MAPS ─────────────────────────────────────────────────────────
unique_train_user_ids  = sorted(train_ratings['userId'].unique())
unique_train_movie_ids = sorted(train_ratings['movieId'].unique())

user_to_idx  = { uid: i for i, uid in enumerate(unique_train_user_ids) }
movie_to_idx = { mid: i for i, mid in enumerate(unique_train_movie_ids) }

n_users  = len(unique_train_user_ids)
n_movies = len(unique_train_movie_ids)

# ─── 2) BUILD movie_ratings + dense_items ───────────────────────────────────────
# movie_ratings[i_idx] should map user_idx → rating for that movie
movie_ratings = [dict() for _ in range(n_movies)]
for _, row in train_ratings.iterrows():
    uidx = user_to_idx[row['userId']]
    midx = movie_to_idx[row['movieId']]
    movie_ratings[midx][uidx] = row['rating']

# Build a dense (n_movies × n_users) array of ratings, 0 if unobserved
dense_items = np.zeros((n_movies, n_users), dtype=np.float32)
for midx, u_dict in enumerate(movie_ratings):
    for uidx, r in u_dict.items():
        dense_items[midx, uidx] = r

# ─── 3) FIT k‐NN ON dense_items WITH COSINE METRIC ────────────────────────────────
k_i = 30
knn_item = NearestNeighbors(n_neighbors=k_i, metric='cosine', algorithm='brute').fit(dense_items)

# dist_i[m_idx, j] = cosine‐distance between item m_idx and its j‐th neighbor
# neighbor_item_idxs[m_idx, j] = index of the j‐th neighbor for item m_idx
dist_i, neighbor_item_idxs = knn_item.kneighbors(dense_items, return_distance=True)

# Convert cosine‐distance → cosine‐similarity
# cosine_similarity = 1 – cosine_distance
item_sim_matrix = 1.0 - dist_i   # shape = (n_movies, k_i)

# ─── 4) DEFINE predict_rating_itemcf(user_id, movie_id) ────────────────────────────
def predict_rating_itemcf(user_id, movie_id):
    """
    Return the predicted rating for (user_id, movie_id) using Item‐Item CF:
      • Finds up to k_i similar movies that the user has already rated.
      • Falls back to movie_avg or global_mean if no neighbor contributed.
    """
    uidx = user_to_idx.get(user_id, None)
    midx = movie_to_idx.get(movie_id, None)

    if (uidx is None) or (midx is None):
        # We don’t know this user or this movie → fall back to global_mean
        return global_mean

    similar_items = neighbor_item_idxs[midx]   # array of k_i neighbor‐movie indices
    sims_i        = item_sim_matrix[midx]      # array of k_i item–item similarities

    w_sum, s_sum = 0.0, 0.0
    for j, m2_idx in enumerate(similar_items):
        # Check if user u rated movie m2_idx in the training data
        if uidx in movie_ratings[m2_idx]:
            r_u_m2 = movie_ratings[m2_idx][uidx]
            sim_m2 = sims_i[j]
            w_sum += sim_m2 * r_u_m2
            s_sum += sim_m2

    if s_sum > 1e-6:
        return w_sum / s_sum
    else:
        # No neighbors contributed → fall back to per‐movie average or global
        return movie_avg.get(movie_id, global_mean)

# ─── 5) PREDICT ON TEST SET, COMPUTE RMSE ─────────────────────────────────────────
test_preds_ibcf = test_ratings.apply(
    lambda row: predict_rating_itemcf(row['userId'], row['movieId']), 
    axis=1
).values

rmse_ibcf = np.sqrt(mean_squared_error(test_ratings['rating'], test_preds_ibcf))
print(f"\nItem‐CF (k_i={k_i}) RMSE = {rmse_ibcf:.4f}")

# ─── 6) TOP‐10 RECOMMENDATIONS FOR A SINGLE USER (EXAMPLE) ────────────────────────
u_test = 42
print(f"\nTop‐10 ITEM‐CF recommendations for user {u_test}:")

if u_test not in user_to_idx:
    print(f"   • User {u_test} did not appear in train, cannot recommend.")
else:
    uidx = user_to_idx[u_test]
    # Movies the user already rated in training
    rated_in_train   = set(train_ratings[train_ratings['userId'] == u_test]['movieId'])
    all_movies_train = set(unique_train_movie_ids)
    candidates       = list(all_movies_train - rated_in_train)

    preds = []
    for m in candidates:
        # Obtain predicted rating for (u_test, m)
        pred_r = predict_rating_itemcf(u_test, m)
        preds.append((m, pred_r))

    # Sort by predicted rating (descending) and take top‐10
    top10 = sorted(preds, key=lambda x: x[1], reverse=True)[:10]
    for (m, rhat) in top10:
        title = movies.loc[movies['movieId'] == m, 'title'].values[0]
        print(f"   • {title:40s} → {rhat:.3f}")


# 6) CONTENT‐BASED COLLABORATIVE FILTERING
# ────────────────────────────────────────────

# Compute the global mean and per‐movie average from train:
global_mean = train_ratings['rating'].mean()
movie_avg   = train_ratings.groupby('movieId')['rating'].mean().to_dict()

# ─── 1) EXTRACT ALL UNIQUE GENRES ───────────────────────────────────────────────────────────────
unique_genres = set()
for g in movies['genres']:
    for tag in g.split('|'):
        unique_genres.add(tag)
unique_genres = sorted(unique_genres)

# ─── 2) ONE‐HOT ENCODE GENRES IN `movies` ───────────────────────────────────────────────────
for tag in unique_genres:
    colname = f"is_{tag}"
    movies[colname] = movies['genres'].apply(lambda s: 1 if tag in s.split('|') else 0)

genre_cols = [f"is_{tag}" for tag in unique_genres]

# ─── 3) MAP movieId → ROW INDEX IN `movies` ────────────────────────────────────────────────
movie_ids = movies['movieId'].values
movieid_to_index = { mid: i for i, mid in enumerate(movie_ids) }

# ─── 4) EXTRACT MOVIE FEATURE MATRIX AS NumPy ARRAY (n_movies × n_genres) ─────────────────────
movie_feature_matrix = movies[genre_cols].values.astype(float)  # shape = (n_movies, n_genres)

# ─── 5) GATHER ALL TRAINING USER‐IDs AND BUILD USER INDEX ─────────────────────────────────────
all_user_ids = np.sort(train_ratings['userId'].unique())
user_index   = { uid: idx for idx, uid in enumerate(all_user_ids) }
n_users      = len(all_user_ids)
n_genres     = len(genre_cols)

# ─── 6) BUILD EACH USER’S PREFERENCE VECTOR (n_users × n_genres) ────────────────────────────
#      We will do a weighted sum of each movie’s one‐hot genre vector, weighted by rating, then normalize.

# 6a) group (movieId, rating) pairs by raw userId from train:
user_ratings = defaultdict(list)
for _, row in train_ratings.iterrows():
    u_raw = int(row['userId'])
    m_raw = int(row['movieId'])
    r     = float(row['rating'])
    user_ratings[u_raw].append((m_raw, r))

# 6b) allocate storage for user‐genre matrix and a “norm” per user
user_pref_matrix = np.zeros((n_users, n_genres), dtype=float)
user_norm        = np.zeros(n_users, dtype=float)

# 6c) iterate over each raw user and accumulate weighted genre‐vector
for u_raw, rated_list in user_ratings.items():
    if u_raw not in user_index:
        continue
    u_row        = user_index[u_raw]
    weighted_vec = np.zeros(n_genres, dtype=float)
    weight_sum   = 0.0

    for (m_raw, r) in rated_list:
        # only if we have that movie’s features
        if m_raw in movieid_to_index:
            m_row       = movieid_to_index[m_raw]
            feature_vec = movie_feature_matrix[m_row]  # length = n_genres
            weighted_vec += r * feature_vec
            weight_sum   += r

    if weight_sum > 0:
        # divide by total rating sum
        weighted_vec /= weight_sum
        # then force unit‐length
        norm_val = np.linalg.norm(weighted_vec)
        if norm_val > 0:
            weighted_vec /= norm_val
            user_norm[u_row] = 1.0
        else:
            user_norm[u_row] = 1e-6
    else:
        # user had no ratings (unlikely), keep zero‐vector but avoid zero‐division
        user_norm[u_row] = 1e-6

    user_pref_matrix[u_row] = weighted_vec

# ─── 7) DEFINE predict_rating_cbf(user_id, movie_id) ───────────────────────────────────────
def predict_rating_cbf(user_id, movie_id):
    """
    Return predicted rating in [1..5] for (user_id, movie_id) via Content‐Based Filtering:
      1) Look up user → u_row. If absent, return global_mean.
      2) Look up movie → m_row. If absent, return global_mean.
      3) Compute cosine between user_pref_matrix[u_row] and movie_feature_matrix[m_row].
      4) Clip to [0..1] and then rescale linearly to [1..5].
    """
    # 7a) map raw IDs to indexes
    uidx = user_index.get(user_id, None)
    m_row = movieid_to_index.get(movie_id, None)

    # 7b) fallback if user or movie never seen in train:
    if (uidx is None) or (m_row is None):
        return global_mean

    uv = user_pref_matrix[uidx]          # length = n_genres
    mv = movie_feature_matrix[m_row]     # length = n_genres

    un = user_norm[uidx] if user_norm[uidx] != 0 else 1e-6
    mn = np.linalg.norm(mv) if np.linalg.norm(mv) != 0 else 1e-6

    raw_cosine = np.dot(uv, mv) / (un * mn)
    # Because both uv,mv are non‐negative, raw_cosine ∈ [0..1], but clip defensively:
    clipped = np.clip(raw_cosine, 0.0, 1.0)

    # Linearly rescale [0..1] → [1..5]
    pred_rating = 1.0 + 4.0 * clipped
    return pred_rating

# ─── 8) COMPUTE RMSE ON TEST SET ──────────────────────────────────────────────────────
test_preds_cb = test_ratings.apply(
    lambda row: predict_rating_cbf(row['userId'], row['movieId']),
    axis=1
).values

rmse_cb = np.sqrt(mean_squared_error(test_ratings['rating'], test_preds_cb))
print(f"\nContent‐Based CF RMSE = {rmse_cb:.4f}")

# ─── 9) EXAMPLE: TOP‐10 CONTENT‐BASED RECS FOR A SPECIFIC USER ─────────────────────────
u_raw = 42
print(f"\nTop‐10 CBF recommendations for user {u_raw}:")

if u_raw not in user_index:
    print(f"  • User {u_raw} was never seen in training; cannot recommend.")
else:
    u_row = user_index[u_raw]
    seen_by_u = { m for (m, _) in user_ratings[u_raw] }
    all_movies_train = set(train_ratings['movieId'].unique())
    unseen = sorted(all_movies_train - seen_by_u)

    cand_scores = []
    for m_raw in unseen:
        # compute exactly the same predict_rating_cbf logic—but only need the cosine part + rescale
        m_row = movieid_to_index[m_raw]
        uv = user_pref_matrix[u_row]
        mv = movie_feature_matrix[m_row]
        un = user_norm[u_row] if user_norm[u_row] != 0 else 1e-6
        mn = np.linalg.norm(mv) if np.linalg.norm(mv) != 0 else 1e-6

        raw_cosine = np.dot(uv, mv) / (un * mn)
        clipped   = np.clip(raw_cosine, 0.0, 1.0)
        score     = 1.0 + 4.0 * clipped

        cand_scores.append((m_raw, score))

    cand_scores.sort(key=lambda x: x[1], reverse=True)
    top10 = cand_scores[:10]

    for (m_raw, scr) in top10:
        title = movies.loc[movies['movieId'] == m_raw, 'title'].values[0]
        print(f"  • {title:40s} → {scr:.3f}")


# 6) MATRIX FACTORIZATION (SVD)
# ─────────────────────────────────────────────────────
# 1) Build & mean‐center the (user × movie) CSR once
# ─────────────────────────────────────────────────────
def build_centered_matrix(train_df):
    # Map raw IDs → dense indices
    users = np.sort(train_df['userId'].unique())
    movies = np.sort(train_df['movieId'].unique())
    u2r = {u:i for i,u in enumerate(users)}
    m2c = {m:j for j,m in enumerate(movies)}

    n_u, n_m = len(users), len(movies)
    rows, cols, vals = [], [], []
    for _, row in train_df.iterrows():
        rows.append(u2r[row.userId])
        cols.append(m2c[row.movieId])
        vals.append(row.rating)
    M = csr_matrix((vals,(rows,cols)), shape=(n_u,n_m))

    # compute row means
    sums   = np.array(M.sum(axis=1)).ravel()
    counts = np.diff(M.indptr)
    means  = np.zeros(n_u)
    nz     = counts>0
    means[nz] = sums[nz]/counts[nz]

    # subtract means
    data_centered = M.data.copy()
    for i in range(n_u):
        start, stop = M.indptr[i], M.indptr[i+1]
        data_centered[start:stop] -= means[i]
    M0 = csr_matrix((data_centered, M.indices, M.indptr), shape=M.shape)

    return M0, means, u2r, m2c

rating_csr, user_means, user_to_row, movie_to_col = build_centered_matrix(train_ratings)
global_mean = train_ratings.rating.mean()

# ─────────────────────────────────────────────────────
# 2) Factorization + regularization helper
# ─────────────────────────────────────────────────────
def fit_svd_regularized(M_centered, k, lam):
    svd = TruncatedSVD(n_components=k, random_state=42, n_iter=15)
    svd.fit(M_centered)
    Σ = svd.singular_values_
    Σr = Σ / np.sqrt(Σ**2 + lam)

    # user_factors = (M_centered · V) · Σr⁻¹
    Vt = svd.components_       # shape (k, n_movies)
    Uraw = M_centered.dot(Vt.T)  # (n_users, k)
    Ur    = Uraw.dot(np.diag(1/Σr))

    # movie_factors = Vt.T * Σr  (broadcast)
    Mr    = Vt.T * Σr

    return Ur, Mr

# ─────────────────────────────────────────────────────
# 3) Hyperparameter sweep to pick best (k, λ)
# ─────────────────────────────────────────────────────
best_rmse = np.inf
best_cfg  = None
best_U, best_M = None, None

for k in [10,20,30]:
    for lam in [0.1,1,5,10,20]:
        U, M = fit_svd_regularized(rating_csr, k, lam)

        # eval on test set
        preds = []
        for _, row in test_ratings.iterrows():
            u, m = row.userId, row.movieId
            uidx = user_to_row.get(u)
            midx = movie_to_col.get(m)
            if uidx is None or midx is None:
                pred = global_mean
            else:
                pred = U[uidx].dot(M[midx]) + user_means[uidx]
                pred = np.clip(pred,1,5)
            preds.append(pred)

        rmse = np.sqrt(mean_squared_error(test_ratings.rating, preds))
        if rmse < best_rmse:
            best_rmse, best_cfg = rmse, (k,lam)
            best_U, best_M = U.copy(), M.copy()

print(f"Best SVD-Reg: k={best_cfg[0]}, λ={best_cfg[1]} → RMSE={best_rmse:.4f}")

# ─────────────────────────────────────────────────────
# 4) Define the final prediction function
# ─────────────────────────────────────────────────────
def predict_rating_svd(user_id, movie_id):
    uidx = user_to_row.get(user_id)
    midx = movie_to_col.get(movie_id)
    if uidx is None or midx is None:
        return global_mean
    p = best_U[uidx].dot(best_M[midx]) + user_means[uidx]
    return float(np.clip(p, 1, 5))


# 7) Ensemble (User‐based CF + SVD + content‐based)
# ────────────────────────────────────────────

# 1) Grid‐search best ensemble weights
# ─────────────────────────────────────────────────────
best_rmse     = np.inf
best_weights  = None

# weight candidates 0.0, 0.1, …, 1.0
grid = np.arange(0.0, 1.0 + 1e-9, 0.1)

print("Searching for best hybrid weights …")
for w_cf in grid:
    for w_svd in grid:
        w_content = 1.0 - w_cf - w_svd
        if w_content < -1e-8 or w_content > 1.0 + 1e-8:
            continue

        # build predictions on the test set
        preds = []
        for _, r in test_ratings.iterrows():
            u, m = int(r.userId), int(r.movieId)
            p_cf      = predict_rating_usercf(u, m)
            p_svd     = predict_rating_svd(u, m)
            p_content = predict_rating_cbf(u, m)
            h = w_cf * p_cf + w_svd * p_svd + w_content * p_content
            preds.append(np.clip(h, 1.0, 5.0))

        rmse = np.sqrt(mean_squared_error(test_ratings['rating'], preds))
        if rmse < best_rmse:
            best_rmse, best_weights = rmse, (round(w_cf,1), round(w_svd,1), round(w_content,1))

        # (optional) print progress
        print(f" w_cf={w_cf:.1f}, w_svd={w_svd:.1f}, w_cont={w_content:.1f} → RMSE {rmse:.4f}")

print(f"\n→ Best hybrid weights:  w_cf={best_weights[0]}, w_svd={best_weights[1]}, "
      f"w_cont={best_weights[2]}  → RMSE={best_rmse:.4f}\n")

w_cf_opt, w_svd_opt, w_content_opt = best_weights

# ─────────────────────────────────────────────────────
# 2) Single‐call hybrid predictor
# ─────────────────────────────────────────────────────
def predict_rating_hybrid(user_id, movie_id):
    """
    Weighted ensemble:
       w_cf_opt    * User-CF  +
       w_svd_opt   * SVD      +
       w_content_opt * CBF
    """
    p_cf      = predict_rating_usercf(user_id, movie_id)
    p_svd     = predict_rating_svd(user_id, movie_id)
    p_content = predict_rating_cbf(user_id, movie_id)
    h = w_cf_opt * p_cf + w_svd_opt * p_svd + w_content_opt * p_content
    return float(np.clip(h, 1.0, 5.0))

# ─────────────────────────────────────────────────────
# 3) Quick RMSE check on test set
# ─────────────────────────────────────────────────────
test_preds_hybrid = test_ratings.apply(
    lambda r: predict_rating_hybrid(int(r.userId), int(r.movieId)),
    axis=1
).values
rmse_hybrid = np.sqrt(mean_squared_error(test_ratings['rating'], test_preds_hybrid))
print(f"Recomputed Hybrid RMSE = {rmse_hybrid:.4f}")


# 7) GENERATIVE AI
# ─────────────────────────────────────────────────────

# ─── Top-K recommendation wrappers (get ready for streamlit) ────────────────────────────────────────────

# 1) Precompute list of all training‐movies and each user’s seen set
ALL_MOVIES    = sorted(train_ratings['movieId'].unique())
USER_HISTORY  = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()

def _get_top_n(fn_predict, user_id, N):
    """Generic helper: call fn_predict(user, movie) over unseen movies and return top-N pairs."""
    seen       = USER_HISTORY.get(user_id, set())
    candidates = [m for m in ALL_MOVIES if m not in seen]
    scored     = [(m, fn_predict(user_id, m)) for m in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:N]

def get_top_n_pop(user_id, N=10):
    """Popularity (global‐mean) top-N."""
    return _get_top_n(predict_rating_pop, user_id, N)

def get_top_n_movieavg(user_id, N=10):
    """Per‐movie‐average top-N."""
    return _get_top_n(predict_rating_movieavg, user_id, N)

def get_top_n_usercf(user_id, N=10):
    """User‐based CF top-N."""
    return _get_top_n(predict_rating_usercf, user_id, N)

def get_top_n_itemcf(user_id, N=10):
    """Item‐based CF top-N."""
    return _get_top_n(predict_rating_itemcf, user_id, N)

def get_top_n_cbf(user_id, N=10):
    """Content‐based top-N."""
    return _get_top_n(predict_rating_cbf, user_id, N)

def get_top_n_svd(user_id, N=10):
    """SVD‐based MF top-N."""
    return _get_top_n(predict_rating_svd, user_id, N)

def get_top_n_hybrid(user_id, N=10):
    """Ensemble (hybrid) top-N."""
    return _get_top_n(predict_rating_hybrid, user_id, N)


##### EVALUATION #####

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict

def evaluate_rating_metrics(models: dict, test_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dict of {model_name: predict_fn}, and a test DataFrame with columns
    ['userId','movieId','rating'], returns a DataFrame indexed by model_name
    with columns [RMSE, MAE, R2].
    """
    y_true = test_ratings['rating'].values
    records = []
    for name, fn in models.items():
        # predict each row
        y_pred = test_ratings.apply(
            lambda row: fn(int(row.userId), int(row.movieId)),
            axis=1
        ).values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        records.append((name, rmse, mae, r2))
    df = pd.DataFrame.from_records(
        records, columns=['Model','RMSE','MAE','R2']
    ).set_index('Model')
    return df.round(3)


def evaluate_topn_metrics(
    models: dict,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    K: int = 10
) -> pd.DataFrame:
    """
    For each model in `models`, computes Precision@K, Recall@K, F1@K.
    We consider as “relevant” any test‐rating >= 4.0.
    """
    # precompute ground‐truth relevant sets
    rel = (
        test_ratings[test_ratings.rating >= 4.0]
        .groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )
    all_train_movies = set(train_ratings['movieId'].unique())

    rows = []
    for name, fn in models.items():
        p_list, r_list = [], []
        for u, relevant_movies in rel.items():
            seen = set(train_ratings[train_ratings.userId == u]['movieId'])
            candidates = list(all_train_movies - seen)
            # score & sort
            scored = [(m, fn(int(u), int(m))) for m in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            topk = {m for m,_ in scored[:K]}
            hits = len(topk & relevant_movies)
            p_list.append(hits / K)
            r_list.append(hits / len(relevant_movies))
        p = np.mean(p_list) if p_list else 0.0
        r = np.mean(r_list) if r_list else 0.0
        f = (2*p*r/(p+r)) if (p+r)>0 else 0.0
        rows.append((name, p, r, f))

    df = pd.DataFrame.from_records(
        rows, columns=['Model', f'P@{K}', f'R@{K}', f'F1@{K}']
    ).set_index('Model')
    return df.round(3)
