import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# ---------------- DATA ----------------
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# ---------------- FUNCTIONS ----------------
def trending_movies():
    top_ids = ratings.groupby('movieId').size().sort_values(ascending=False).head(5).index
    return movies[movies['movieId'].isin(top_ids)]

def explore_new(user_id):
    seen = user_movie_matrix.loc[user_id]
    seen_movies = seen[seen > 0].index
    unseen = movies[~movies['movieId'].isin(seen_movies)]
    return unseen.sample(5)

def recommend(user_id):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6].index
    similar_ratings = user_movie_matrix.loc[similar_users]
    mean_ratings = similar_ratings.mean(axis=0)

    user_seen = user_movie_matrix.loc[user_id]
    mean_ratings = mean_ratings[user_seen == 0]

    top_ids = mean_ratings.sort_values(ascending=False).head(10).index

    recs = movies[movies['movieId'].isin(top_ids)]
    return recs

# ---------------- UI ----------------
st.title("🎬 Mini Movie AI Recommender")

user_id = st.number_input("Enter User ID", min_value=1, max_value=1000, step=1)

if st.button("Get Recommendations"):

    st.subheader("🔥 Top Picks For You")
    recs = recommend(user_id)
    st.write(recs[['title']].head(5))

    st.subheader("📈 Trending Now")
    st.write(trending_movies()[['title']])

    st.subheader("🌍 Explore New Movies")
    st.write(explore_new(user_id)[['title']])