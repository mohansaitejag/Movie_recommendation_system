

import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd

PROFILE_FILE = "user_profiles.json"
MATRIX_FILE  = "user_movie_matrix.csv"

print("=" * 55)
print("         CineAgent AI Movie Recommender")
print("=" * 55)
print(" Loading dataset...")

ratings = pd.read_csv("ratings.csv")
movies  = pd.read_csv("movies.csv")
df = pd.merge(ratings, movies, on="movieId")[
    ['userId', 'movieId', 'rating', 'title', 'genres']
]
print(f" Movies:{df['title'].nunique()}  Users:{df['userId'].nunique()}  Ratings:{len(df)}")

print(" Building user-movie matrix...")
user_movie_matrix = df.pivot_table(
    index='userId', columns='title', values='rating'
).fillna(0)

if os.path.exists(MATRIX_FILE):          # merge persisted feedback so ratings survive restarts
    saved = pd.read_csv(MATRIX_FILE, index_col=0)
    saved.index = saved.index.astype(int)
    for col in saved.columns:
        if col not in user_movie_matrix.columns: user_movie_matrix[col] = 0.0
    for idx in saved.index:
        if idx not in user_movie_matrix.index:   user_movie_matrix.loc[idx] = 0.0
    user_movie_matrix.update(saved)
    print(" Saved feedback loaded.")

# Precompute once to avoid repeated groupby on every request
_popularity = (
    df.groupby('title')['rating'].agg(['mean', 'count'])
    .query('count > 50').sort_values('mean', ascending=False)
)
_movie_genres = (
    df[['title', 'genres']].drop_duplicates('title')
    .set_index('title')['genres'].apply(lambda x: set(x.split('|')))
)
_movie_avg = df.groupby('title')['rating'].mean()


# ---- TECHNIQUE 1: User-Based CF -- Manual Cosine Similarity --------------
#
# cos(u,v) = (u·v) / (||u|| × ||v||)
# L2-normalise every row, then similarity = single matrix multiply.
# No Python loop over users -- fully vectorised.

def cosine_sim_matrix(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10      # guard: empty user vector -> no similarity
    normed = mat / norms
    return normed @ normed.T       # (n×m) @ (m×n) = (n×n) similarity matrix

print(" Computing user similarity (manual cosine)...")
user_sim    = cosine_sim_matrix(user_movie_matrix.values)
user_sim_df = pd.DataFrame(
    user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index
)
print(" Ready!\n")


# ---- TECHNIQUE 2: Item-Based CF -- Manual Pearson Correlation ------------
#
# r(a,b) = Σ(ai−ā)(bi−b̄) / √(Σ(ai−ā)² × Σ(bi−b̄)²)
#
# Co-rated mask: only users who rated BOTH items contribute.
# Without this, zeros pollute the mean and distort the correlation.
# Minimum 3 co-raters: fewer makes the value unreliable.

def pearson_item_sim(u, v):
    mask = (u > 0) & (v > 0)
    if mask.sum() < 3: return 0.0
    a, b = u[mask], v[mask]
    num  = np.sum((a - a.mean()) * (b - b.mean()))
    den  = np.sqrt(np.sum((a - a.mean())**2) * np.sum((b - b.mean())**2))
    return 0.0 if den == 0 else float(num / den)

def recommend_similar_items(user_id, profile, n=50):
    # score(candidate) = Σ pearson(seed, candidate) × user_rating(seed)
    rated = get_all_rated(user_id, profile)
    if user_id not in user_movie_matrix.index: return []
    row   = user_movie_matrix.loc[user_id]
    liked = row[row >= 4].index.tolist() or row[row > 0].nlargest(3).index.tolist()
    if not liked: return []
    top_movies = df['title'].value_counts().head(200).index
    sub    = user_movie_matrix[[c for c in top_movies if c in user_movie_matrix.columns]]
    scores = {}
    for seed in liked[:5]:
        if seed not in sub.columns: continue
        sv = sub[seed].values
        for movie in sub.columns:
            if movie in rated or movie == seed: continue
            sim = pearson_item_sim(sv, sub[movie].values)
            if sim > 0: scores[movie] = scores.get(movie, 0) + sim * row[seed]
    return sorted(scores, key=scores.get, reverse=True)[:n]


# ---- PERSISTENCE ---------------------------------------------------------

def save_matrix(): user_movie_matrix.to_csv(MATRIX_FILE)

def rebuild_sim():
    # Rebuilds cosine similarity after feedback.
    # O(n²) in user count -- acceptable for MovieLens 100K (~940 users).
    global user_sim, user_sim_df
    user_sim    = cosine_sim_matrix(user_movie_matrix.values)
    user_sim_df = pd.DataFrame(
        user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index
    )

def load_profiles():
    return json.load(open(PROFILE_FILE)) if os.path.exists(PROFILE_FILE) else {}

def save_profiles(p):
    json.dump(p, open(PROFILE_FILE, "w"), indent=2)

DEFAULT_WEIGHTS = {"collab": 0.45, "item_cf": 0.25, "genre": 0.20, "top": 0.10}

def get_profile(user_id, profiles):
    key = str(user_id)
    if key not in profiles:
        profiles[key] = {
            "liked_genres": [], "disliked_genres": [], "disliked_movies": [],
            "rated_movies": [], "session_count": 0, "engine_weights": DEFAULT_WEIGHTS.copy()
        }
    p = profiles[key]
    if "engine_weights" not in p: p["engine_weights"] = DEFAULT_WEIGHTS.copy()
    if "rated_movies"   not in p: p["rated_movies"]   = []
    return p

def is_cold_start(user_id):
    # Fewer than 5 ratings in the matrix means CF has no useful signal yet
    if user_id not in user_movie_matrix.index: return True
    return int((user_movie_matrix.loc[user_id] > 0).sum()) < 5


# ---- RECOMMENDATION HELPERS ----------------------------------------------

def get_all_rated(user_id, profile):
    # Union of matrix-rated + profile-rated: double-safety so a disliked
    # movie can't reappear through either path after feedback.
    matrix_rated = set()
    if user_id in user_movie_matrix.index:
        row = user_movie_matrix.loc[user_id]
        matrix_rated = set(row[row > 0].index)
    return matrix_rated | set(profile.get("rated_movies", []))

def get_top_movies(user_id, profile, n=80):
    rated = get_all_rated(user_id, profile)
    return [t for t in _popularity.index if t not in rated][:n]

def recommend_by_genre(user_id, ref_movie, profile, n=80):
    rated = get_all_rated(user_id, profile)
    if ref_movie not in _movie_genres: return []
    gs = _movie_genres[ref_movie]
    scores = {t: (len(gs & g), _movie_avg.get(t, 0))
              for t, g in _movie_genres.items()
              if t not in rated and t != ref_movie and len(gs & g) >= 2}
    return sorted(scores, key=lambda t: scores[t], reverse=True)[:n]

def recommend_users(user_id, profile, n=80):
    rated = get_all_rated(user_id, profile)
    if user_id not in user_sim_df.index: return get_top_movies(user_id, profile, n)
    similar = user_sim_df[user_id].sort_values(ascending=False)[1:6]
    scores  = {}
    for su, ss in similar.items():
        for movie, rating in user_movie_matrix.loc[su].items():
            if movie not in rated and rating > 0:
                scores[movie] = scores.get(movie, 0) + ss * rating
    return sorted(scores, key=scores.get, reverse=True)[:n]

def get_movies_by_genre(user_id, genre, profile, n=80):
    rated = get_all_rated(user_id, profile)
    return [t for t, g in _movie_genres.items() if genre in g and t not in rated][:n]


# ---- TECHNIQUE 3: Utility Scorer (Multi-Factor Ranking) ------------------
#
# Four signals combined into a single [0,1] score:
#   g_score -- movie matches the user's chosen genre?
#   m_score -- genre overlap with current mood
#   r_score -- community avg rating normalised to [0,1]
#   c_score -- weighted sum of similar-user ratings, capped at 1
#
# With a reference movie: ref_score (content similarity) gets 40% weight.
# Without: genre + mood split the top weight instead.

def utility_score(title, preferred_genre, mood_genres, collab_scores, ref_genres=None, liked_genres=None):
    if title not in _movie_genres: return 0.0
    genres  = _movie_genres[title]
    avg_r   = _movie_avg.get(title, 0)
    g_score = 1.0 if preferred_genre and preferred_genre in genres else 0.0
    if liked_genres and not g_score:
        overlap = len(genres & set(liked_genres))
        g_score = min(overlap / 2.0, 1.0)
    m_score = (min(len(genres & set(mood_genres)) / max(len(mood_genres), 1), 1.0)
               if mood_genres else 0.0)
    r_score = (avg_r - 0.5) / 4.5                           # map [0.5,5] -> [0,1]
    c_score = min(collab_scores.get(title, 0) / 25.0, 1.0)
    if ref_genres:
        ref_s = min(len(genres & set(ref_genres)) / max(len(ref_genres), 1), 1.0)
        return round(g_score*0.15 + m_score*0.15 + ref_s*0.40 + r_score*0.20 + c_score*0.10, 4)
    return round(g_score*0.35 + m_score*0.25 + r_score*0.25 + c_score*0.15, 4)


MOOD_SUPPRESS = {
    "Happy": ["Horror"],  "Sad": ["Horror", "Action", "Thriller"],
    "Excited": [],        "Scared": ["Comedy", "Romance"],
    "Thoughtful": ["Horror"],  "Any": []
}


# ---- SMART RECOMMEND (Perceive -> Reason -> Act) -------------------------

def smart_recommend(user_id, mood, mood_genres, preferred_genre, ref_movie, profile, n=15):
    pool, engine_used = {}, {}
    cold = is_cold_start(user_id)

    # PERCEIVE: gather candidates from up to 4 sources, tag each with its engine
    if cold:
        for t in get_top_movies(user_id, profile):
            pool[t], engine_used[t] = "Popular pick -- rate movies to personalise", "top"
    else:
        for t in recommend_users(user_id, profile):          # Technique 1
            pool[t], engine_used[t] = "Users like you watched this", "collab"
        for t in recommend_similar_items(user_id, profile):  # Technique 2
            if t not in pool:
                pool[t], engine_used[t] = "Similar to movies you rated highly", "item_cf"
        if ref_movie:
            for t in recommend_by_genre(user_id, ref_movie, profile):
                if t not in pool:
                    pool[t], engine_used[t] = f"Because you liked {ref_movie}", "genre"
        if preferred_genre:
            for t in get_movies_by_genre(user_id, preferred_genre, profile):
                if t not in pool:
                    pool[t], engine_used[t] = f"Matches your genre pick: {preferred_genre}", "genre"
        if profile.get("liked_genres"):
            for lg in profile["liked_genres"][:3]:
                for t in get_movies_by_genre(user_id, lg, profile, n=20):
                    if t not in pool:
                        pool[t], engine_used[t] = f"Matches a genre you like: {lg}", "genre"
        if len(pool) < 20:
            for t in get_top_movies(user_id, profile):
                if t not in pool:
                    pool[t], engine_used[t] = "Top rated overall", "top"

    # REASON: filter by mood suppression and dislikes; relax progressively if pool is thin
    all_rated       = get_all_rated(user_id, profile)
    suppress        = set(MOOD_SUPPRESS.get(mood, []))
    disliked_genres = set(profile.get("disliked_genres", []))
    disliked_movies = set(profile.get("disliked_movies", []))

    def passes(title, strict):
        if title in all_rated or title in disliked_movies: return False
        genres = _movie_genres.get(title)
        if genres is None or genres & (suppress | disliked_genres): return False
        if strict and mood_genres and not cold:
            if not (genres & set(mood_genres)): return False
        return True

    filtered = {t: s for t, s in pool.items() if passes(t, strict=True)}
    if len(filtered) < 10:
        filtered = {t: s for t, s in pool.items() if passes(t, strict=False)}
    if len(filtered) < 10:
        filtered = pool

    collab_scores = {}
    if not cold and user_id in user_sim_df.index:
        for su, ss in user_sim_df[user_id].sort_values(ascending=False)[1:6].items():
            for movie, rating in user_movie_matrix.loc[su].items():
                if movie not in all_rated and rating > 0:
                    collab_scores[movie] = collab_scores.get(movie, 0) + ss * rating

    ref_genres = list(_movie_genres[ref_movie]) if ref_movie and ref_movie in _movie_genres else []

    # ACT: score each candidate with utility function × per-user engine trust weight
    ew = profile.get("engine_weights", DEFAULT_WEIGHTS)
    scored = []
    for title, source in filtered.items():
        base = utility_score(title, preferred_genre, mood_genres, collab_scores, ref_genres, profile.get("liked_genres", []))
        eng  = engine_used.get(title, "top")
        w    = round(base * (1.0 + ew.get(eng, 0.1)), 4)   # Technique 4 trust boost
        scored.append((title, " | ".join(_movie_genres.get(title, set())),
                       round(_movie_avg.get(title, 0), 1), w, source, eng))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:n]


# ---- TECHNIQUE 4: Online Engine Weight Learning --------------------------
#
# engine_weights is a probability distribution over the 4 engines per user.
# loved/liked  -> weight += 0.08 (reward the engine that found this movie)
# disliked     -> weight -= 0.08 (penalise it)
# Weights are renormalised to sum to 1 after every update.
# lr=0.08 means ~10 consecutive positives are needed to double an engine's weight.

def handle_feedback(user_id, movie_title, feedback, profile, profiles, engine=None):
    genres = list(_movie_genres.get(movie_title, set()))
    if movie_title not in user_movie_matrix.columns: user_movie_matrix[movie_title] = 0.0
    if user_id    not in user_movie_matrix.index:    user_movie_matrix.loc[user_id] = 0.0

    rating_map = {"loved": 5.0, "liked": 4.0, "disliked": 1.0, "watched": 3.0, 
                  "5": 5.0, "4": 4.0, "3": 3.0, "2": 2.0, "1": 1.0}
    fb_str = str(feedback)
    if fb_str in rating_map:
        user_movie_matrix.loc[user_id, movie_title] = rating_map[fb_str]

    if fb_str in ("loved", "liked", "5", "4"):
        for g in genres:
            if g not in profile["liked_genres"]:    profile["liked_genres"].append(g)
            if g     in profile["disliked_genres"]: profile["disliked_genres"].remove(g)
    elif fb_str in ("disliked", "1", "2", "3"):
        for g in genres:
            if g not in profile["disliked_genres"]: profile["disliked_genres"].append(g)
        if movie_title not in profile["disliked_movies"]:
            profile["disliked_movies"].append(movie_title)

    if fb_str != "skip":
        if movie_title not in profile["rated_movies"]:
            profile["rated_movies"].append(movie_title)
        if engine and engine in profile["engine_weights"]:
            w  = profile["engine_weights"]
            lr = 0.08
            if   fb_str in ("loved", "liked", "5", "4"): w[engine] = min(0.80, w[engine] + lr)
            elif fb_str in ("disliked", "1", "2", "3"):       w[engine] = max(0.05, w[engine] - lr)
            total = sum(w.values())
            profile["engine_weights"] = {k: round(v/total, 3) for k, v in w.items()}
        rebuild_sim()
        save_matrix()
    save_profiles(profiles)


# ---- DISPLAY HELPERS -----------------------------------------------------

def divider(): print("  " + "-" * 51)
def header(t): print(f"\n  {'='*51}\n   {t}\n  {'='*51}")

def show_movies(recs, mood, preferred_genre):
    tag = f"Mood: {mood}" + (f"  |  Genre: {preferred_genre}" if preferred_genre else "")
    header(f"Top Picks  --  {tag}")
    for i, (title, genres, avg_r, score, source, eng) in enumerate(recs, 1):
        t = title[:40] + "..." if len(title) > 40 else title
        print(f"\n   {i}.  {t}")
        print(f"       Rating  : {'★'*round(avg_r)}{'☆'*(5-round(avg_r))}  {avg_r}/5.0")
        print(f"       Genres  : {genres}")
        print(f"       Why     : {source}  [{eng}]")
    divider()

def ask_feedback(movie_title):
    t = movie_title[:42] + "..." if len(movie_title) > 42 else movie_title
    print(f"\n   Movie : {t}")
    print("   Rate  : [L] Loved  [G] Liked  [D] Disliked  [W] Watched  [S] Skip")
    opts = {
        "l": ("loved",    "Loved! Recommending similar ones next."),
        "g": ("liked",    "Good choice noted."),
        "d": ("disliked", "Understood. Filtering similar ones out."),
        "w": ("watched",  "Marked as watched."),
        "s": ("skip",     "Skipped.")
    }
    fb, msg = opts.get(input("   --> ").strip().lower(), ("skip", "Skipped."))
    print(f"       {msg}")
    return fb

def show_summary(log, profile):
    header("Session Summary")
    counts = Counter(e["fb"] for e in log)
    print(f"\n   Loved:{counts['loved']}  Liked:{counts['liked']}  "
          f"Disliked:{counts['disliked']}  Watched:{counts['watched']}")
    if profile["liked_genres"]:
        print(f"   You enjoy  : {', '.join(profile['liked_genres'][:6])}")
    print(f"   Total rated : {len(profile.get('rated_movies',[]))} movies across all sessions")
    ew = profile.get("engine_weights", {})
    if ew:
        print("\n   Engine trust weights (learned from your feedback):")
        for eng, w in sorted(ew.items(), key=lambda x: x[1], reverse=True):
            print(f"     {eng:<10} {'█'*int(w*20)} {w:.2f}")
    print("\n   Profile saved. Better picks next session!")


# ---- MOODS + GENRES + PREFERENCES ----------------------------------------

MOODS = {
    "1": ("Happy",      ["Comedy", "Animation", "Romance"]),
    "2": ("Sad",        ["Drama", "Romance"]),
    "3": ("Excited",    ["Action", "Adventure", "Sci-Fi"]),
    "4": ("Scared",     ["Horror", "Thriller"]),
    "5": ("Thoughtful", ["Documentary", "Drama", "Mystery"]),
    "6": ("Any",        [])
}
GENRES = ["Action","Adventure","Animation","Comedy","Crime","Documentary",
          "Drama","Fantasy","Horror","Mystery","Romance","Sci-Fi","Thriller","Western"]

def ask_preferences():
    header("What are we watching tonight?")
    print("\n   How are you feeling?")
    for k, (label, _) in MOODS.items(): print(f"   [{k}] {label}")
    mood_label, mood_genres = MOODS.get(input("\n   Pick (1-6) : ").strip(), ("Any", []))
    print("\n   Preferred genre? (Enter to skip)")
    for i, g in enumerate(GENRES, 1):
        print(f"   [{i:2}] {g}", end=("   " if i % 3 != 0 else "\n"))
    print()
    g_in = input("   Pick number : ").strip()
    preferred_genre = GENRES[int(g_in)-1] if g_in.isdigit() and 1 <= int(g_in) <= len(GENRES) else None
    ref_movie = input("\n   A movie you recently liked? (Enter to skip)\n   Title : ").strip() or None
    return mood_label, mood_genres, preferred_genre, ref_movie


# ---- MAIN AGENT LOOP -----------------------------------------------------

def run_agent(user_id):
    profiles = load_profiles()
    profile  = get_profile(user_id, profiles)
    session_log, liked_count, GOAL = [], 0, 3
    profile["session_count"] = profile.get("session_count", 0) + 1
    save_profiles(profiles)

    header(f"Welcome!  User: {user_id}  |  Session #{profile['session_count']}")
    if profile["liked_genres"]:  print(f"\n   I remember you like : {', '.join(profile['liked_genres'][:5])}")
    if profile.get("rated_movies"): print(f"   Movies rated so far : {len(profile['rated_movies'])}")
    if is_cold_start(user_id):
        print("\n   New user -- showing popular movies first.")
        print("   Rate a few to unlock personalised recommendations.\n")

    while True:
        mood, mood_genres, preferred_genre, ref_movie = ask_preferences()
        print("\n   Finding your movies...")
        recs = smart_recommend(user_id, mood, mood_genres, preferred_genre, ref_movie, profile, n=15)
        if not recs:
            print("\n   No matches found. Try different preferences.\n"); continue
        show_movies(recs, mood, preferred_genre)
        print("\n   Rate each one so I can learn your taste:"); divider()
        for rec in recs:
            title, *_, eng = rec
            fb = ask_feedback(title)
            handle_feedback(user_id, title, fb, profile, profiles, engine=eng)
            session_log.append({"movie": title, "fb": fb, "engine": eng})
            if fb in ("loved", "liked"): liked_count += 1
            if liked_count >= GOAL:
                print(f"\n   Goal reached! Liked {liked_count} movies. Better picks next session.")
                break
        divider()
        if input("\n   More recommendations? (y/n) : ").strip().lower() != "y": break
        liked_count = 0
    show_summary(session_log, profile)


# ---- EVALUATION METRICS --------------------------------------------------
#
# Precision@K : did the held-out liked movie appear in the top-K list?
# Coverage    : fraction of the full catalogue ever recommended.
# Diversity   : avg unique genres per recommendation list.
#
# Holdout: random sample from ratings >= 4 per user.
# Random sample is less biased than always picking the single top-rated movie.

def evaluate_metrics(k=10, n_users=30):
    print(f"\n{'='*60}\n  EVALUATION METRICS  (K={k}, test users={n_users})\n{'='*60}")
    test_users = df.groupby('userId').size().pipe(lambda s: s[s >= 10]).index.tolist()[:n_users]
    hits, total, all_recs, div_list = 0, 0, set(), []
    profiles = load_profiles()

    for uid in test_users:
        liked_rows = df[(df['userId'] == uid) & (df['rating'] >= 4)]
        if liked_rows.empty: continue
        holdout  = liked_rows.sample(1, random_state=uid)['title'].values[0]
        original = None
        if holdout in user_movie_matrix.columns:
            original = user_movie_matrix.loc[uid, holdout]
            user_movie_matrix.loc[uid, holdout] = 0.0
        recs   = smart_recommend(uid, "Any", [], None, None, get_profile(uid, profiles), n=k)
        titles = [r[0] for r in recs]
        if holdout in titles: hits += 1
        total += 1
        all_recs.update(titles)
        div_list.append(len(set(g for r in recs for g in _movie_genres.get(r[0], set()))))
        if original is not None: user_movie_matrix.loc[uid, holdout] = original

    total_movies = df['title'].nunique()
    precision = hits / total if total else 0
    coverage  = len(all_recs) / total_movies if total_movies else 0
    diversity = sum(div_list) / len(div_list) if div_list else 0
    print(f"  Precision@{k}        : {precision:.3f}  ({hits}/{total} -- holdout in top {k})")
    print(f"  Catalogue coverage  : {coverage:.3f}  ({len(all_recs)}/{total_movies} movies)")
    print(f"  Avg genre diversity : {diversity:.1f} unique genres per list")
    print("=" * 60 + "\n")


# ---- TEST SCENARIOS -------------------------------------------------------

def _h(n, title):     # small helper to print scenario headers consistently
    print(f"\n{'='*60}\n  SCENARIO {n}: {title}\n{'='*60}")

def run_test_scenarios():
    profiles = load_profiles()

    # Scenario 1: new user should get popularity-only recommendations
    _h(1, "Cold Start -- new user, no rating history")
    cp = get_profile(99999, profiles)
    assert is_cold_start(99999), "FAIL: 99999 should be cold start"
    recs = smart_recommend(99999, "Any", [], None, None, cp, n=10)
    assert len(recs) > 0, "FAIL: cold start returned nothing"
    engines = {r[5] for r in recs}
    assert engines == {"top"}, f"FAIL: expected only 'top' engine, got {engines}"
    assert all("Popular" in r[4] for r in recs), "FAIL: source should mention popularity"
    print(f"  Got {len(recs)} recs, engines={engines}")
    for i, r in enumerate(recs[:5], 1): print(f"  {i}. {r[0]}")
    print("  PASS: popularity-only path, correct engine tag")

    # Scenario 2: warm user should use CF engines; mood suppression must hold
    _h(2, "Warm user + mood filter (Excited/Action)")
    warm_uid = int(df.groupby('userId').size().pipe(lambda s: s[s >= 15]).index[0])
    wp       = get_profile(warm_uid, profiles)
    assert not is_cold_start(warm_uid), f"FAIL: {warm_uid} should not be cold start"
    recs = smart_recommend(warm_uid, "Excited", ["Action", "Adventure"], None, None, wp, n=10)
    assert len(recs) > 0, "FAIL: warm user returned nothing"
    engines = {r[5] for r in recs}
    # at least one CF engine must appear; fallback to 'top' alone would indicate a bug
    assert engines & {"collab", "item_cf"}, \
        f"FAIL: expected collab or item_cf in engines, got {engines}"
    suppress = set(MOOD_SUPPRESS.get("Excited", []))
    for r in recs:
        assert not (_movie_genres.get(r[0], set()) & suppress), \
            f"FAIL: '{r[0]}' has suppressed genre for Excited mood"
    print(f"  userId={warm_uid}, engines={engines}")
    for i, r in enumerate(recs[:5], 1): print(f"  {i}. {r[0]} [{r[5]}]")
    print("  PASS: CF engine present, mood suppression respected")

    # Scenario 3: dislike removes the movie and penalises the correct engine
    _h(3, "Feedback loop -- dislike removes movie, engine weight drops")
    tp       = get_profile(88888, profiles)
    recs_pre = smart_recommend(88888, "Any", [], None, None, tp, n=20)
    assert recs_pre, "FAIL: no initial recommendations"
    target, t_eng  = recs_pre[0][0], recs_pre[0][5]
    w_before       = dict(tp["engine_weights"])
    print(f"  Target: '{target}'  engine: {t_eng}  weights: {w_before}")
    handle_feedback(88888, target, "disliked", tp, profiles, engine=t_eng)
    recs_post = smart_recommend(88888, "Any", [], None, None, tp, n=20)
    assert target not in [r[0] for r in recs_post], f"FAIL: '{target}' still in recs after dislike"
    assert target in tp["disliked_movies"], "FAIL: not added to disliked_movies"
    assert tp["engine_weights"][t_eng] < w_before[t_eng], \
        f"FAIL: engine '{t_eng}' weight should have dropped"
    print(f"  Weights after: {tp['engine_weights']}")
    print(f"  '{target}' correctly absent.  PASS")

    for uid in ["99999", "88888"]: profiles.pop(uid, None)
    save_profiles(profiles)
    print(f"\n{'='*60}\n  All 3 scenarios PASSED.\n{'='*60}\n")


# ---- ENTRY POINT ---------------------------------------------------------

if __name__ == "__main__":
    if   "--test"    in sys.argv: run_test_scenarios()
    elif "--metrics" in sys.argv: evaluate_metrics(k=10, n_users=30)
    else:
        header("CineAgent -- AI Movie Recommender")
        uid = input("\n   Enter your User ID (number) : ").strip()
        run_agent(int(uid) if uid.isdigit() else 999)