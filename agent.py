"""
========================================================
  CineAgent — AI Movie Recommender Agent
  Built on: MovieLens 100K Dataset
  Agent Type: Learning Agent (Goal-Based + Utility-Based)

  FIXES APPLIED:
  1. Matrix is now saved to disk after every feedback
  2. Matrix is loaded + merged at startup so ratings persist
  3. Watched movies (including disliked) are never shown again
  4. New users get a clean profile automatically
========================================================
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────
#  FILE PATHS
# ──────────────────────────────────────────────────────

PROFILE_FILE = "user_profiles.json"
MATRIX_FILE  = "user_movie_matrix.csv"   # ← NEW: persistent matrix

# ──────────────────────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────────────────────

print("=" * 55)
print("         CineAgent AI Movie Recommender")
print("=" * 55)
print(" Loading dataset...")

ratings = pd.read_csv("ratings.csv")
movies  = pd.read_csv("movies.csv")
df      = pd.merge(ratings, movies, on="movieId")
df      = df[['userId', 'movieId', 'rating', 'title', 'genres']]

print(f" Movies  : {df['title'].nunique()}")
print(f" Users   : {df['userId'].nunique()}")
print(f" Ratings : {len(df)}")

print("\n Building user-movie matrix...")
user_movie_matrix = df.pivot_table(
    index='userId', columns='title', values='rating'
).fillna(0)

# ── FIX 1: Merge saved matrix so previous feedback survives restarts ──
if os.path.exists(MATRIX_FILE):
    print(" Loading saved feedback matrix...")
    saved_matrix = pd.read_csv(MATRIX_FILE, index_col=0)

    # Convert index to int to match userId type
    saved_matrix.index = saved_matrix.index.astype(int)

    # Add any new columns from saved matrix that aren't in current matrix
    for col in saved_matrix.columns:
        if col not in user_movie_matrix.columns:
            user_movie_matrix[col] = 0.0

    # Add any new user rows from saved matrix
    for idx in saved_matrix.index:
        if idx not in user_movie_matrix.index:
            user_movie_matrix.loc[idx] = 0.0

    # Overwrite cells where saved_matrix has a non-zero rating
    user_movie_matrix.update(saved_matrix)
    print(" Saved feedback loaded successfully!")

print(" Computing user similarity...")
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)
print(" Ready!\n")

# ──────────────────────────────────────────────────────
#  MATRIX PERSISTENCE  (NEW)
# ──────────────────────────────────────────────────────

def save_matrix():
    """Save the full user-movie matrix to disk."""
    user_movie_matrix.to_csv(MATRIX_FILE)

# ──────────────────────────────────────────────────────
#  USER PROFILE  (persistent memory across sessions)
# ──────────────────────────────────────────────────────

def load_profiles():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

def get_profile(user_id, profiles):
    key = str(user_id)
    if key not in profiles:
        profiles[key] = {
            "liked_genres":    [],
            "disliked_genres": [],
            "disliked_movies": [],
            "rated_movies":    [],   # ← FIX 2: track ALL rated movies to avoid repeats
            "session_count":   0
        }
    # Migrate old profiles that don't have rated_movies key
    if "rated_movies" not in profiles[key]:
        profiles[key]["rated_movies"] = []
    return profiles[key]

# ──────────────────────────────────────────────────────
#  REBUILD SIMILARITY after every feedback
# ──────────────────────────────────────────────────────

def rebuild_similarity():
    global user_similarity, user_similarity_df
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

# ──────────────────────────────────────────────────────
#  RECOMMENDATION FUNCTIONS
# ──────────────────────────────────────────────────────

def get_watched(user_id):
    """
    Returns all movies the user has interacted with in the matrix.
    FIX 3: Includes movies rated 1.0 (disliked) so they never reappear.
    Previously only checked rating > 0, but 1.0 is also > 0 so this was
    already correct — the real bug was the matrix resetting on restart.
    Now that the matrix persists, this correctly excludes all rated movies.
    """
    if user_id not in user_movie_matrix.index:
        return set()
    row = user_movie_matrix.loc[user_id]
    return set(row[row > 0].index)

def get_all_rated(user_id, profile):
    """
    Combined watched set: matrix-based + profile-based rated_movies list.
    This is the double-safety net so nothing slips through.
    """
    return get_watched(user_id) | set(profile.get("rated_movies", []))

def get_top_movies(user_id, profile, n=50):
    rated = get_all_rated(user_id, profile)
    avg   = df.groupby('title')['rating'].mean()
    count = df.groupby('title')['rating'].count()
    top   = pd.DataFrame({'avg': avg, 'count': count})
    top   = top[top['count'] > 50].sort_values('avg', ascending=False)
    top   = top[~top.index.isin(rated)]
    return top.head(n).index.tolist()

def recommend_by_genre(user_id, movie_name, profile, n=50):
    rated = get_all_rated(user_id, profile)
    row   = df[df['title'] == movie_name]
    if row.empty:
        return []
    genre_set = set(row.iloc[0]['genres'].split('|'))
    temp = df.copy()
    temp['score'] = temp['genres'].apply(
        lambda x: len(genre_set & set(x.split('|')))
    )
    recs = temp[(temp['score'] >= 2) & (temp['title'] != movie_name)]
    recs = recs[~recs['title'].isin(rated)]
    return recs.sort_values(['score', 'rating'], ascending=False)['title'].drop_duplicates().head(n).tolist()

def recommend_users(user_id, profile, n=50):
    rated = get_all_rated(user_id, profile)
    if user_id not in user_similarity_df.index:
        return get_top_movies(user_id, profile, n)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    scores = {}
    for sim_user, sim_score in similar_users.items():
        for movie, rating in user_movie_matrix.loc[sim_user].items():
            if movie not in rated and rating > 0:
                scores[movie] = scores.get(movie, 0) + sim_score * rating
    return [m for m, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:n]

def get_movies_by_genre(user_id, genre, profile, n=50):
    rated = get_all_rated(user_id, profile)
    g     = df[df['genres'].str.contains(genre, case=False)]
    g     = g[~g['title'].isin(rated)]
    top   = g.groupby('title')['rating'].mean().sort_values(ascending=False)
    return top.head(n).index.tolist()

# ──────────────────────────────────────────────────────
#  UTILITY SCORE  (ranks the combined pool)
# ──────────────────────────────────────────────────────

def utility_score(title, preferred_genre, mood_genres, collab_scores):
    rows = df[df['title'] == title]
    if rows.empty:
        return 0.0
    genres       = set(rows.iloc[0]['genres'].split('|'))
    avg_r        = rows['rating'].mean()
    genre_score  = 1.0 if preferred_genre and preferred_genre in genres else 0.0
    mood_score   = min(len(genres & set(mood_genres)) / max(len(mood_genres), 1), 1.0)
    rating_score = (avg_r - 0.5) / 4.5
    collab_score = min(collab_scores.get(title, 0) / 25.0, 1.0)
    return round(genre_score*0.35 + mood_score*0.25 + rating_score*0.25 + collab_score*0.15, 4)

# ──────────────────────────────────────────────────────
#  MOOD RULES  (suppress genres based on mood)
# ──────────────────────────────────────────────────────

MOOD_SUPPRESS = {
    "Happy":      ["Horror"],
    "Sad":        ["Horror", "Action", "Thriller"],
    "Excited":    [],
    "Scared":     ["Comedy", "Romance"],
    "Thoughtful": ["Horror"],
    "Any":        []
}

# ──────────────────────────────────────────────────────
#  SMART RECOMMEND  (combines all 3 methods + scoring)
# ──────────────────────────────────────────────────────

def smart_recommend(user_id, mood, mood_genres, preferred_genre, ref_movie, profile, n=15):

    # Build pool from all 3 methods — track source of each movie
    pool = {}   # title → source label

    for t in recommend_users(user_id, profile, n=80):
        pool[t] = "Users like you watched this"

    if ref_movie:
        for t in recommend_by_genre(user_id, ref_movie, profile, n=80):
            if t not in pool:
                pool[t] = f"Because you liked {ref_movie}"

    if preferred_genre:
        for t in get_movies_by_genre(user_id, preferred_genre, profile, n=80):
            if t not in pool:
                pool[t] = f"Top rated in {preferred_genre}"

    if len(pool) < 20:
        for t in get_top_movies(user_id, profile, n=80):
            if t not in pool:
                pool[t] = "Top rated overall"

    # Remove disliked movies
    for t in list(pool):
        if t in set(profile.get("disliked_movies", [])):
            del pool[t]

    # Remove ALL previously rated movies (double-safety)
    all_rated = get_all_rated(user_id, profile)
    for t in list(pool):
        if t in all_rated:
            del pool[t]

    # Apply mood + disliked genre rules
    suppress      = set(MOOD_SUPPRESS.get(mood, []))
    disliked      = set(profile.get("disliked_genres", []))
    pool_filtered = {}
    for t, source in pool.items():
        row = df[df['title'] == t]
        if row.empty:
            continue
        genres = set(row.iloc[0]['genres'].split('|'))
        if genres & (suppress | disliked):
            continue
        pool_filtered[t] = source

    if len(pool_filtered) < 10:
        pool_filtered = pool   # relax rules if too strict

    # Build collab scores for utility function
    collab_scores = {}
    if user_id in user_similarity_df.index:
        sim_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
        rated     = get_all_rated(user_id, profile)
        for sim_user, sim_score in sim_users.items():
            for movie, rating in user_movie_matrix.loc[sim_user].items():
                if movie not in rated and rating > 0:
                    collab_scores[movie] = collab_scores.get(movie, 0) + sim_score * rating

    # Score and rank
    scored = []
    for title, source in pool_filtered.items():
        score      = utility_score(title, preferred_genre, mood_genres, collab_scores)
        row        = df[df['title'] == title].iloc[0]
        avg_r      = df[df['title'] == title]['rating'].mean()
        genres_str = row['genres'].replace('|', ' | ')
        scored.append((title, genres_str, round(avg_r, 1), score, source))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:n]

# ──────────────────────────────────────────────────────
#  FEEDBACK HANDLER  (updates matrix + profile + disk)
# ──────────────────────────────────────────────────────

def handle_feedback(user_id, movie_title, feedback, profile, profiles):
    row    = df[df['title'] == movie_title]
    genres = row.iloc[0]['genres'].split('|') if not row.empty else []

    # Ensure the movie column and user row exist in the matrix
    if movie_title not in user_movie_matrix.columns:
        user_movie_matrix[movie_title] = 0.0
    if user_id not in user_movie_matrix.index:
        user_movie_matrix.loc[user_id] = 0.0

    # Map feedback to numeric rating and update matrix
    rating_map = {"loved": 5.0, "liked": 4.0, "disliked": 1.0, "watched": 3.0}
    if feedback in rating_map:
        user_movie_matrix.loc[user_id, movie_title] = rating_map[feedback]

    # Update liked/disliked genres in profile
    if feedback in ("loved", "liked"):
        for g in genres:
            if g not in profile["liked_genres"]:
                profile["liked_genres"].append(g)
            if g in profile["disliked_genres"]:
                profile["disliked_genres"].remove(g)

    elif feedback == "disliked":
        for g in genres:
            if g not in profile["disliked_genres"]:
                profile["disliked_genres"].append(g)
        if movie_title not in profile["disliked_movies"]:
            profile["disliked_movies"].append(movie_title)

    # FIX 2: Track every rated movie so it's never recommended again
    if feedback != "skip":
        if movie_title not in profile["rated_movies"]:
            profile["rated_movies"].append(movie_title)
        rebuild_similarity()   # agent gets smarter after every feedback
        save_matrix()          # ← FIX 1: persist matrix to disk

    save_profiles(profiles)

# ──────────────────────────────────────────────────────
#  DISPLAY HELPERS  (clean readable terminal output)
# ──────────────────────────────────────────────────────

def divider():
    print("  " + "-" * 51)

def header(text):
    print("\n  " + "=" * 51)
    print(f"   {text}")
    print("  " + "=" * 51)

def show_movies(recs, mood, preferred_genre):
    title_line = f"Top 15 Picks   Mood: {mood}"
    if preferred_genre:
        title_line += f"   Genre: {preferred_genre}"
    header(title_line)

    for i, (title, genres, avg_r, score, source) in enumerate(recs, 1):
        t      = title if len(title) <= 40 else title[:37] + "..."
        filled = round(avg_r)
        stars  = "★" * filled + "☆" * (5 - filled)
        print(f"\n   {i}.  {t}")
        print(f"       Rating  : {stars}  {avg_r}/5.0")
        print(f"       Genres  : {genres}")
        print(f"       Why     : {source}")

    divider()

def ask_feedback_for(movie_title):
    t = movie_title if len(movie_title) <= 42 else movie_title[:39] + "..."
    print(f"\n   Movie : {t}")
    print("   Rate  : [L] Loved   [G] Liked   [D] Disliked   [W] Watched   [S] Skip")
    choice = input("   → ").strip().lower()

    options = {
        "l": ("loved",    "❤️  Loved! Recommending similar ones next."),
        "g": ("liked",    "👍 Good choice noted."),
        "d": ("disliked", "👎 Understood. Filtering similar ones out."),
        "w": ("watched",  "📝 Marked as watched."),
        "s": ("skip",     "⏭  Skipped.")
    }
    feedback, message = options.get(choice, ("skip", "⏭  Skipped."))
    print(f"       {message}")
    return feedback

def show_summary(log, profile):
    header("Session Summary")
    loved    = sum(1 for e in log if e["fb"] == "loved")
    liked    = sum(1 for e in log if e["fb"] == "liked")
    disliked = sum(1 for e in log if e["fb"] == "disliked")

    print(f"\n   ❤️  Loved    : {loved} movie(s)")
    print(f"   👍 Liked    : {liked} movie(s)")
    print(f"   👎 Disliked : {disliked} movie(s)")

    if profile["liked_genres"]:
        print(f"\n   You enjoy  : {', '.join(profile['liked_genres'][:6])}")
    if profile["disliked_genres"]:
        print(f"   Avoiding   : {', '.join(profile['disliked_genres'][:6])}")

    total_rated = len(profile.get("rated_movies", []))
    print(f"\n   Total movies rated across all sessions : {total_rated}")
    print("\n   Profile saved. Better picks next session!")
    print("  " + "=" * 51 + "\n")

# ──────────────────────────────────────────────────────
#  MOOD + GENRE MENUS
# ──────────────────────────────────────────────────────

MOODS = {
    "1": ("Happy",      ["Comedy", "Animation", "Romance"]),
    "2": ("Sad",        ["Drama", "Romance"]),
    "3": ("Excited",    ["Action", "Adventure", "Sci-Fi"]),
    "4": ("Scared",     ["Horror", "Thriller"]),
    "5": ("Thoughtful", ["Documentary", "Drama", "Mystery"]),
    "6": ("Any",        [])
}

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "Western"
]

def ask_preferences():
    header("What are we watching tonight?")

    # Mood
    print("\n   How are you feeling?")
    for k, (label, _) in MOODS.items():
        print(f"   [{k}] {label}")
    mood_key             = input("\n   Pick (1-6) : ").strip()
    mood_label, mood_genres = MOODS.get(mood_key, ("Any", []))

    # Genre
    print("\n   Preferred genre? (Enter to skip)")
    for i, g in enumerate(GENRES, 1):
        print(f"   [{i:2}] {g}", end="   " if i % 3 != 0 else "\n")
    print()
    g_input = input("   Pick number : ").strip()
    preferred_genre = None
    if g_input.isdigit() and 1 <= int(g_input) <= len(GENRES):
        preferred_genre = GENRES[int(g_input) - 1]

    # Reference movie
    print("\n   A movie you recently liked? (Enter to skip)")
    ref_movie = input("   Title : ").strip() or None

    return mood_label, mood_genres, preferred_genre, ref_movie

# ──────────────────────────────────────────────────────
#  MAIN AGENT LOOP
# ──────────────────────────────────────────────────────

def run_agent(user_id):
    profiles    = load_profiles()
    profile     = get_profile(user_id, profiles)
    session_log = []
    liked_count = 0
    GOAL        = 3   # session goal: user likes at least 3 movies

    profile["session_count"] = profile.get("session_count", 0) + 1
    save_profiles(profiles)

    header(f"Welcome!   User: {user_id}   Session #{profile['session_count']}")

    if profile["liked_genres"]:
        print(f"\n   I remember you like : {', '.join(profile['liked_genres'][:5])}")
    if profile.get("rated_movies"):
        print(f"   Movies rated so far : {len(profile['rated_movies'])}")

    while True:

        # 1. PERCEIVE — ask user preferences
        mood, mood_genres, preferred_genre, ref_movie = ask_preferences()

        # 2. REASON — run all 3 methods + score
        print("\n   Finding your movies... ")
        recs = smart_recommend(user_id, mood, mood_genres, preferred_genre, ref_movie, profile, n=15)

        if not recs:
            print("\n   No matches found. Try different preferences.\n")
            continue

        # 3. ACT — display movies
        show_movies(recs, mood, preferred_genre)

        # 4. FEEDBACK — one movie at a time
        print("\n   Rate each one so I can learn your taste:")
        divider()

        for title, genres, avg_r, score, source in recs:
            fb = ask_feedback_for(title)
            handle_feedback(user_id, title, fb, profile, profiles)
            session_log.append({"movie": title, "fb": fb})

            if fb in ("loved", "liked"):
                liked_count += 1

            # Goal check
            if liked_count >= GOAL:
                print(f"\n   🏆 Goal reached! You liked {liked_count} movies.")
                print("   Next session recommendations will be even better.")
                break

        # 5. CONTINUE?
        divider()
        again = input("\n   Want more recommendations? (y / n) : ").strip().lower()
        if again != "y":
            break
        liked_count = 0

    # SESSION END
    show_summary(session_log, profile)

# ──────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    header("CineAgent  —  AI Movie Recommender")
    uid = input("\n   Enter your User ID (number) : ").strip()
    try:
        user_id = int(uid)
    except ValueError:
        user_id = 999
    run_agent(user_id)