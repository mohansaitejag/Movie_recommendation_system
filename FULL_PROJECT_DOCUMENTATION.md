# 🎬 AIKR (AI-powered Movie Recommender) - Full Project Documentation

This document serves as the master reference for the AIKR project. It covers the entire system architecture, the machine learning algorithms, the frontend UI implementation, and instructions on how to run it.

---

## 1. Project Architecture
The project is divided into two main components that communicate via a REST API:
1. **Backend (Python/Flask)**: Handles data loading, TF-IDF vectorization, Cosine Similarity calculations, and serves recommendations as JSON.
2. **Frontend (HTML/CSS/JS)**: A highly polished, Netflix-style web interface that fetches data from the backend and dynamically renders it using Vanilla JavaScript and CSS Grid/Flexbox.

---

## 2. The Machine Learning Backend (`movie.py`)

The backend is responsible for the actual "AI" part of the application. It uses the MovieLens dataset (`movies.csv`, `ratings.csv`).

### A. Data Loading & Preprocessing
When the server starts, it loads the datasets using Pandas:
```python
import pandas as pd
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
```

### B. The Recommendation Engine
We use **Content-Based Filtering** (Genre Similarity) to find movies similar to what the user has watched.

1. **TF-IDF Vectorization**: We convert the text genres (e.g., "Action|Adventure|Sci-Fi") into mathematical vectors.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))
```

2. **Cosine Similarity**: We calculate the mathematical angle between these vectors to find how similar two movies are. A score of 1.0 means exact match.
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)
```

### C. The API Endpoint
The Flask app exposes an endpoint `GET /recommend/<username>`. It maps usernames (like `sri`, `alex`, `john`) to user IDs (1, 2, 3), looks up what movies they rated highly (>= 4 stars), and uses the Cosine Similarity matrix to find 4 similar movies for each watched movie.

*Note: The server runs on port `5001` to avoid conflicts with macOS AirPlay services which run on port `5000`.*

---

## 3. The Premium Frontend (`index.html`)

The frontend was built without heavy frameworks to maintain performance, relying on modern CSS and Vanilla JS.

### A. Styling & UI Design
*   **Glassmorphism**: The navigation bar uses `backdrop-filter: blur(10px)` to create a frosted glass effect.
*   **Cinematic Gradients**: Because we lack actual movie posters, we dynamically generate deep, radial CSS gradients.
*   **Film Grain Overlay**: An SVG noise filter is applied over the cards (`mix-blend-mode: overlay`) to simulate premium film grain.
*   **Horizontal Carousels**: We use `overflow-x: auto` and hide the scrollbar (`::-webkit-scrollbar { display: none; }`) to create swipeable rows.

### B. API Integration (JavaScript)
The frontend dynamically fetches the data from the Flask API and injects HTML directly into the page.

```javascript
function login() {
    let user = document.getElementById("username").value || 'sri';
    
    // Fetch from the Flask backend
    fetch("http://127.0.0.1:5001/recommend/" + user)
    .then(res => res.json())
    .then(data => {
        // 1. Inject Top Picks
        const topPicksHTML = data.top_picks.map((m, i) =>
            `<div class="card grad-${(i%5)+1}"><div class="placeholder-title">${m}</div></div>`
        ).join("");
        document.getElementById("top-picks-container").innerHTML = topPicksHTML;

        // 2. Inject "Because You Watched" Rows
        let becauseHTML = "";
        data.because.forEach((b, idx) => {
            becauseHTML += `<h3>Because you watched ${b.because}:</h3><div class="cards-row">`;
            b.recommendations.forEach((r, i) => {
                becauseHTML += `<div class="byw-card grad-${(i%5)+1}"><div class="placeholder-title">${r}</div></div>`;
            });
            becauseHTML += `</div>`;
        });
        document.getElementById("because-container").innerHTML = becauseHTML;
    })
    .catch(err => console.error(err));
}
```

---

## 4. How to Run the Project

To launch the full application, you need two terminal windows.

### Terminal 1: Start the Backend (Data Engine)
The backend needs to load the datasets and compute the similarity matrix.
```bash
cd AIKR
python movie.py
```
*(Wait until you see it say `Running on http://127.0.0.1:5001`)*

### Terminal 2: Serve the Frontend
To avoid browser security blocks when opening files directly, serve the HTML via a local Python server:
```bash
cd AIKR
python -m http.server 8000
```

### Final Step: View the App
1. Open your web browser (Chrome, Safari, etc.).
2. Go to **`http://localhost:8000`**.
3. In the top right corner, enter `sri`, `alex`, or `john`.
4. Click **Login**.
*(Note: The very first login fetch might take ~30 seconds as the backend computes the machine learning matrix in memory. Subsequent clicks will be instant).*

---
*Developed as a comprehensive, full-stack Machine Learning implementation.*
