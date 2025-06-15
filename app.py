import os
import streamlit as st
import pandas as pd

from recommender import (
    get_top_n_pop,
    get_top_n_movieavg,
    get_top_n_usercf,
    get_top_n_itemcf,
    get_top_n_cbf,
    get_top_n_svd,
    get_top_n_hybrid,
    predict_rating_hybrid
)

# ─── define paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "DATA", "ml-latest-small")

# ─── cached loaders ───────────────────────────────────────────
@st.cache_data
def load_movies():
    path = os.path.join(DATA_DIR, "movies.csv")
    return pd.read_csv(path)

@st.cache_data
def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.csv")
    return pd.read_csv(path)

movies  = load_movies()
ratings = load_ratings()

# ─── sidebar ──────────────────────────────────────────────────
st.sidebar.title("MovieLens Recommender")

users = sorted(ratings.userId.unique())
selected_user  = st.sidebar.selectbox("Select User ID", users)

model_options = [
    "Popularity (Global Mean)",
    "Movie Average",
    "User-CF",
    "Item-CF",
    "Content-Based",
    "SVD",
    "Hybrid"
]
selected_model = st.sidebar.selectbox("Recommendation Model", model_options)

N = st.sidebar.slider("Number of Recommendations (Top N)", 5, 20, 10)


# ─── main ─────────────────────────────────────────────────────
st.title(f"Top {N} Recommendations for User {selected_user}")

if selected_model == "Popularity (Global Mean)":
    recs = get_top_n_pop(selected_user, N)
elif selected_model == "Movie Average":
    recs = get_top_n_movieavg(selected_user, N)
elif selected_model == "User-CF":
    recs = get_top_n_usercf(selected_user, N)
elif selected_model == "Item-CF":
    recs = get_top_n_itemcf(selected_user, N)
elif selected_model == "Content-Based":
    recs = get_top_n_cbf(selected_user, N)
elif selected_model == "SVD":
    recs = get_top_n_svd(selected_user, N)
else:
    recs = get_top_n_hybrid(selected_user, N)

if recs:
    df = pd.DataFrame(recs, columns=["movieId", "pred_rating"])
    df = df.merge(movies[["movieId", "title"]], on="movieId")
    df = df[["title", "pred_rating"]]
    st.table(df)
else:
    st.write("No recommendations available for this user.")

# ─── rating predictor ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.write("### Predict a Single Rating")
movie_to_pred = st.sidebar.number_input("Movie ID", min_value=1, step=1)
if st.sidebar.button("Predict"):
    pred = predict_rating_hybrid(selected_user, movie_to_pred)
    title = movies.loc[movies.movieId == movie_to_pred, "title"].squeeze() \
            if movie_to_pred in movies.movieId.values else "Unknown"
    st.sidebar.write(f"Predicted rating for **{title}**: {pred:.2f}")
