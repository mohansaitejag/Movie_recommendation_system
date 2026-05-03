from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import agentd

app = Flask(__name__)
CORS(app)

TMDB_API_KEY = "50f9182da7839079f6340a94c326b250"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"

poster_cache = {}

def get_poster(title):
    if title in poster_cache:
        return poster_cache[title]
    try:
        clean_title = title.split(" (")[0]
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title, "page": 1}
        res = requests.get(url, params=params, timeout=3).json()
        results = res.get("results", [])
        poster = None
        if results and results[0].get("poster_path"):
            poster = TMDB_POSTER_BASE + results[0]["poster_path"]
        poster_cache[title] = poster
        return poster
    except Exception:
        return None

@app.route("/poster/<path:title>")
def api_poster(title):
    poster = get_poster(title)
    return jsonify({"poster": poster})

@app.route("/login", methods=["POST"])
def login_api():
    data = request.json
    user_id = data.get("user_id")
    try:
        user_id = int(user_id)
        if user_id in agentd.user_movie_matrix.index:
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "User ID not found in dataset"})
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Invalid User ID format"})

@app.route("/search/<int:user_id>")
def api_search(user_id):
    q = request.args.get("q", "").strip()
    profiles = agentd.load_profiles()
    profile = agentd.get_profile(user_id, profiles)
    
    if not q:
        return jsonify(agentd.get_top_movies(user_id, profile, 12))
    
    df = agentd.df
    result = df[df['title'].str.contains(q, case=False, na=False)]
    rated = agentd.get_all_rated(user_id, profile)
    result = result[~result['title'].isin(rated)]
    
    return jsonify(result['title'].drop_duplicates().head(12).tolist())

@app.route("/recommend_movie/<int:user_id>")
def api_recommend_movie(user_id):
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"searched_movie": "", "recommendations": []})
    
    profiles = agentd.load_profiles()
    profile = agentd.get_profile(user_id, profiles)
    
    recs = agentd.recommend_by_genre(user_id, q, profile, 6)
    return jsonify({
        "searched_movie": q,
        "recommendations": recs
    })

@app.route("/genre/<int:user_id>/<genre>")
def api_genre(user_id, genre):
    profiles = agentd.load_profiles()
    profile = agentd.get_profile(user_id, profiles)
    if genre.lower() == 'all':
        recs = agentd.get_top_movies(user_id, profile, 18)
    else:
        recs = agentd.get_movies_by_genre(user_id, genre, profile, 18)
    return jsonify(recs)

@app.route("/feedback", methods=["POST"])
def api_feedback():
    data = request.json
    user_id = int(data.get("user_id"))
    movie_title = data.get("movie_title")
    feedback = data.get("feedback")
    
    profiles = agentd.load_profiles()
    profile = agentd.get_profile(user_id, profiles)
    
    agentd.handle_feedback(user_id, movie_title, feedback, profile, profiles)
    
    recs = agentd.smart_recommend(user_id, "Any", [], None, movie_title, profile, 20)
    rec_titles = [r[0] for r in recs]
    
    return jsonify({
        "success": True,
        "recommendations": rec_titles
    })

@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    profiles = agentd.load_profiles()
    profile = agentd.get_profile(user_id, profiles)

    mood = request.args.get('mood', 'Any')
    MOOD_GENRES_MAP = {
        "Action": ["Action"],
        "Comedy": ["Comedy"],
        "Crime": ["Crime"],
        "Drama": ["Drama"],
        "Romance": ["Romance"],
        "Thriller": ["Thriller"],
        "Any": []
    }
    mood_genres = MOOD_GENRES_MAP.get(mood, [])

    recs = agentd.smart_recommend(user_id, mood, mood_genres, None, None, profile, 20)
    top_picks_titles = [r[0] for r in recs]
    
    because_results = []
    try:
        top_1_liked = []
        if user_id in agentd.user_movie_matrix.index:
            user_ratings = agentd.user_movie_matrix.loc[user_id]
            liked_movies = set(user_ratings[user_ratings >= 4.0].index.tolist())
            
            # get the single most recently liked movie
            for m in reversed(profile.get("rated_movies", [])):
                if m in liked_movies:
                    top_1_liked.append(m)
                    break
        
        # Fallback to original dataset logic if they haven't liked anything recently
        if not top_1_liked:
            user_data = agentd.df[agentd.df["userId"] == user_id]
            watched_ids = user_data[user_data["rating"] >= 4]["movieId"]
            watched_movies = agentd.df[agentd.df["movieId"].isin(watched_ids)].drop_duplicates('title')
            top_1_liked = watched_movies.head(1)["title"].tolist()
            
        for title in top_1_liked:
            recs = agentd.recommend_by_genre(user_id, title, profile, 20)
            because_results.append({
                "because": title,
                "recommendations": recs
            })
    except Exception as e:
        print("Because error:", e)
        
    users_like_you_titles = agentd.recommend_users(user_id, profile, 20)
    
    return jsonify({
        "top_picks": top_picks_titles,
        "because": because_results,
        "users_like_you": users_like_you_titles
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)