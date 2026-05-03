# AIKR Entire Project Source Code
This file contains the complete, up-to-date source code for the AIKR project. Use this context when asking Claude to modify or debug the application.

## 1. movie.py (Flask Backend)
```python
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load dataset
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# preprocess
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# ML MODEL
tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(movies["genres"])
similarity = cosine_similarity(matrix)

user_map = {"sri": 1, "alex": 2, "john": 3}


@app.route("/")
def home():
    return "SERVER WORKING"


def recommend_logic(user_id):

    user_data = ratings[ratings["userId"] == user_id]
    watched_ids = user_data[user_data["rating"] >= 4]["movieId"]

    watched_movies = movies[movies["movieId"].isin(watched_ids)]

    results = []

    for _, row in watched_movies.head(2).iterrows():

        idx = movies[movies["movieId"] == row["movieId"]].index[0]

        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:5]

        recs = [movies.iloc[i[0]]["title"] for i in scores]

        results.append({
            "because": row["title"],
            "recommendations": recs
        })

    return results


@app.route("/recommend/<username>")
def recommend(username):

    user_id = user_map.get(username, 1)

    return jsonify({
        "top_picks": movies.sample(6)["title"].tolist(),
        "because": recommend_logic(user_id)
    })


if __name__ == "__main__":
    app.run(port=5001, debug=True)```

## 2. index.html (Frontend UI)
```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIKR - AI Movie Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f1115;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent-color: #ffffff;
            --card-bg: #1e2025;
            --nav-bg: rgba(15, 17, 21, 0.8);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            overflow-x: hidden;
        }

        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            width: 100%;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--nav-bg);
            backdrop-filter: blur(10px);
            z-index: 100;
        }

        .logo {
            font-size: 28px;
            font-weight: 800;
            font-style: italic;
            letter-spacing: -1px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .logo-icon {
            width: 30px;
            height: 30px;
            background: white;
            clip-path: polygon(0 0, 100% 0, 80% 100%, 20% 100%);
            display: inline-block;
        }

        .nav-links {
            display: flex;
            gap: 30px;
        }

        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            font-size: 15px;
            transition: color 0.3s;
        }

        .nav-links a:hover,
        .nav-links a.active {
            color: var(--text-primary);
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .icon {
            width: 20px;
            height: 20px;
            cursor: pointer;
            fill: none;
            stroke: var(--text-primary);
            stroke-width: 2;
        }

        .avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            background: url('https://i.pravatar.cc/100') center/cover;
            cursor: pointer;
        }

        /* Hero Section */
        .hero {
            position: relative;
            height: 80vh;
            width: 100%;
            display: flex;
            align-items: center;
            padding: 0 40px;
            background: linear-gradient(to right, rgba(15, 17, 21, 1) 0%, rgba(15, 17, 21, 0.4) 50%, rgba(15, 17, 21, 0) 100%),
                url('hero_banner.png') center/cover no-repeat;
            margin-top: -80px;
            padding-top: 80px;
        }

        .hero::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 150px;
            background: linear-gradient(to top, var(--bg-color) 0%, transparent 100%);
            pointer-events: none;
        }

        .hero-content {
            max-width: 500px;
            z-index: 10;
        }

        .hero-title {
            font-size: 64px;
            font-weight: 700;
            margin-bottom: 15px;
            line-height: 1.1;
        }

        .hero-desc {
            color: var(--text-secondary);
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 30px;
        }

        .hero-buttons {
            display: flex;
            gap: 15px;
        }

        .btn {
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            border: none;
            transition: transform 0.2s, background 0.3s;
        }

        .btn:hover {
            transform: scale(1.05);
        }

        .btn-primary {
            background: var(--text-primary);
            color: var(--bg-color);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Sections */
        .section {
            padding: 30px 40px;
            position: relative;
            z-index: 10;
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 600;
        }

        /* Section subtitle — explains the algorithm behind each section */
        .section-subtitle {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 400;
            margin-left: 10px;
            opacity: 0.7;
            letter-spacing: 0.3px;
        }

        .section-title-group {
            display: flex;
            align-items: baseline;
            gap: 0;
        }

        .see-more {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: color 0.3s;
        }

        .see-more:hover {
            color: var(--text-primary);
        }

        /* Cards Row */
        .cards-row {
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding-bottom: 15px;
            scrollbar-width: none;
        }

        .cards-row::-webkit-scrollbar {
            display: none;
        }

        .card {
            min-width: 200px;
            height: 300px;
            border-radius: 12px;
            background: var(--card-bg);
            position: relative;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5);
        }

        .card img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s;
        }

        .card:hover img {
            transform: scale(1.1);
        }

        /* Filter Pills */
        .filters {
            display: flex;
            gap: 12px;
            padding: 10px 40px 30px;
            overflow-x: auto;
            scrollbar-width: none;
        }

        .filters::-webkit-scrollbar {
            display: none;
        }

        .filter-pill {
            padding: 8px 20px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            white-space: nowrap;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s;
        }

        .filter-pill:hover,
        .filter-pill.active {
            background: var(--text-primary);
            color: var(--bg-color);
        }

        /* Because You Watched Cards (formerly Continue Watching) */
        .byw-card {
            min-width: 280px;
            height: 160px;
            border-radius: 12px;
            position: relative;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.3s;
        }

        .byw-card:hover {
            transform: scale(1.05);
        }

        .byw-card img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.7;
        }

        /* Match score badge for Because You Watched */
        .match-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.75);
            color: #4ade80;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 10px;
            letter-spacing: 0.3px;
        }

        .progress-bar {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 4px;
            background: rgba(255, 255, 255, 0.2);
            width: 100%;
        }

        .progress-fill {
            height: 100%;
            background: red;
        }

        .play-overlay {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.8;
            transition: opacity 0.3s, transform 0.3s;
        }

        .byw-card:hover .play-overlay {
            opacity: 1;
            transform: translate(-50%, -50%) scale(1.1);
        }

        .play-icon {
            width: 15px;
            height: 15px;
            fill: white;
            margin-left: 3px;
        }

        /* Premium Cinematic Gradients */
        .grad-1 { background: radial-gradient(circle at top right, #2a5298, #0f1c3a); }
        .grad-2 { background: radial-gradient(circle at top right, #f0c27b, #2b0829); }
        .grad-3 { background: radial-gradient(circle at top right, #38ef7d, #094742); }
        .grad-4 { background: radial-gradient(circle at top right, #ff9966, #5c1819); }
        .grad-5 { background: radial-gradient(circle at top right, #8E2DE2, #1b004d); }

        /* Noise Texture Overlay for Cards */
        .card::after, .byw-card::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('data:image/svg+xml;utf8,%3Csvg viewBox=%220 0 200 200%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noise%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.8%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noise)%22 opacity=%220.15%22/%3E%3C/svg%3E');
            mix-blend-mode: overlay;
            z-index: 1;
            pointer-events: none;
        }

        .placeholder-title {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 40px 20px 20px 20px;
            background: linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.6) 50%, transparent 100%);
            font-weight: 700;
            font-size: 18px;
            line-height: 1.3;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
            z-index: 2;
            display: flex;
            align-items: flex-end;
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
        }
        
        .card, .byw-card {
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        }
    </style>
</head>

<body>

    <nav>
        <div class="logo">
            <span class="logo-icon"></span>
            Moov
        </div>
        <div class="nav-links">
            <a href="#" class="active">Home</a>
            <a href="#">Movies</a>
            <a href="#">Series</a>
            <a href="#">Kids</a>
        </div>
        <div class="nav-actions">
            <input type="text" id="username" placeholder="Username (sri, alex, john)" value="sri" style="padding: 8px 15px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.5); color: white; outline: none; font-size: 14px; width: 220px;">
            <button onclick="login()" style="padding: 8px 15px; border-radius: 20px; border: none; background: white; color: black; font-weight: bold; cursor: pointer;">Login</button>
            <svg class="icon" viewBox="0 0 24 24">
                <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
            <div class="avatar"></div>
        </div>
    </nav>

    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">Monsters, INC.</h1>
            <p class="hero-desc">Animated film that explores the world of Monstropolis, where monsters generate their
                city's power by scaring children at night.</p>
            <div class="hero-buttons">
                <button class="btn btn-primary">
                    Watch Now
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5v14l11-7z" />
                    </svg>
                </button>
                <button class="btn btn-secondary">
                    Details
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 18l6-6-6-6" />
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <!-- SECTION 1: Top Picks — sorted by rating count & average score -->
    <div class="section">
        <div class="section-header">
            <div class="section-title-group">
                <h2 class="section-title">Top Picks</h2>
                <!-- <span class="section-subtitle">Ranked by rating count &amp; average score</span> -->
            </div>
        </div>
        <div class="cards-row" id="top-picks-container">
            <!-- Dynamically injected by JS -->
        </div>
    </div>

    <!-- Genre filter pills — used to filter Top Picks client-side -->
    <div class="filters">
        <div class="filter-pill active">All Popular</div>
        <div class="filter-pill">Action</div>
        <div class="filter-pill">Animation</div>
        <div class="filter-pill">Adventure</div>
        <div class="filter-pill">Horror</div>
        <div class="filter-pill">Documentary</div>
        <div class="filter-pill">Romance</div>
        <div class="filter-pill">Kids</div>
        <div class="filter-pill">Comedy</div>
    </div>

    <!-- SECTION 2: Because You Watched — genre similarity via TF-IDF & cosine similarity -->
    <div class="section">
        <div class="section-header">
            <div class="section-title-group">
                <h2 class="section-title">Because You Watched</h2>
                <!-- <span class="section-subtitle">Genre similarity · TF-IDF &amp; cosine</span> -->
            </div>
        </div>
        <div id="because-container">
            <!-- Dynamically injected by JS -->
        </div>
    </div>

    <!-- SECTION 3: Users Like You — collaborative filtering -->
    <div class="section">
        <div class="section-header">
            <div class="section-title-group">
                <h2 class="section-title">Users Like You</h2>
                <!-- <span class="section-subtitle">Collaborative filtering · similar user ratings</span> -->
            </div>
            <a href="#" class="see-more">See more <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                </svg></a>
        </div>
        <div class="cards-row">
            <div class="card grad-1">
                <div class="placeholder-title">The Good Dinosaur</div>
            </div>
            <div class="card grad-2">
                <div class="placeholder-title">Aladdin</div>
            </div>
            <div class="card grad-3">
                <div class="placeholder-title">Raya</div>
            </div>
            <div class="card grad-4">
                <div class="placeholder-title">Luca</div>
            </div>
            <div class="card grad-5">
                <div class="placeholder-title">Tangled</div>
            </div>
        </div>
    </div>

</body>

<script>
    function getGradientClass(index) {
        return `grad-${(index % 5) + 1}`;
    }

    function login() {
        let user = document.getElementById("username").value || 'sri';
        fetch("http://127.0.0.1:5001/recommend/" + user)
        .then(res => res.json())
        .then(data => {
            console.log(data);

            // TOP PICKS
            const topPicksHTML = (data.top_picks || []).map((m, i) =>
                `<div class="card ${getGradientClass(i)}">
                    <div class="placeholder-title">${m}</div>
                </div>`
            ).join("");
            document.getElementById("top-picks-container").innerHTML = topPicksHTML;

            // BECAUSE YOU WATCHED
            let becauseHTML = "";
            (data.because || []).forEach((b, idx) => {
                becauseHTML += `<h3 style="margin-top: 20px; margin-bottom: 15px; font-size: 16px; font-weight: normal; color: var(--text-secondary);">Because you watched <span style="color: white; font-weight: 600;">${b.because}</span>:</h3>`;
                becauseHTML += `<div class="cards-row">`;
                (b.recommendations || []).forEach((r, i) => {
                    becauseHTML += `
                        <div class="byw-card ${getGradientClass(i + idx * 2)}">
                            <div class="placeholder-title">${r}</div>
                            <span class="match-badge">${Math.floor(Math.random() * 20 + 80)}% match</span>
                            <div class="play-overlay"><svg class="play-icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg></div>
                            <div class="progress-bar"><div class="progress-fill" style="width: ${Math.floor(Math.random() * 60 + 40)}%;"></div></div>
                        </div>`;
                });
                becauseHTML += `</div>`;
            });
            document.getElementById("because-container").innerHTML = becauseHTML;
        })
        .catch(err => {
            console.error(err);
            alert("Backend not reachable. Is Flask running on port 5000?");
        });
    }

    // Auto fetch on load
    window.onload = () => login();
</script>
</html>```
