# AIKR Movie Recommender

An AI-powered movie recommendation system built with Python, Flask, and a web frontend. This project uses the MovieLens dataset to provide personalized movie recommendations based on user ratings, genre similarity, and collaborative filtering.

## 🎬 Project Overview

AIKR (AI Movie Recommender) is a web application that helps users discover movies through intelligent recommendations. The system analyzes user rating patterns and movie genres to suggest films that match individual preferences.

### Key Features

- **Personalized Recommendations**: Get movie suggestions based on your rating history
- **Genre-Based Similarity**: Recommendations powered by TF-IDF vectorization and cosine similarity
- **Collaborative Filtering**: Find movies liked by similar users
- **Top Picks**: Popular movies based on rating count and average score
- **Web Interface**: Clean, responsive UI with movie cards and sections
- **REST API**: Flask backend serving JSON recommendations

## 📊 Dataset

This project uses the **MovieLens Latest Small Dataset** from GroupLens Research:

- **100,836 ratings** from **610 users** on **9,742 movies**
- **3,683 tag applications** across movies
- Data collected between March 29, 1996 - September 24, 2018
- Files: `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`

### Data Structure

- **movies.csv**: movieId, title, genres
- **ratings.csv**: userId, movieId, rating, timestamp
- **tags.csv**: userId, movieId, tag, timestamp
- **links.csv**: movieId, imdbId, tmdbId

## 🏗️ Architecture

### Backend (Flask API)
- **Framework**: Flask with CORS support
- **Machine Learning**:
  - TF-IDF Vectorizer for genre similarity
  - Cosine similarity for content-based recommendations
  - User-user collaborative filtering
- **Endpoints**:
  - `GET /` - Health check
  - `GET /recommend/<username>` - Get recommendations for user

### Frontend (HTML/CSS/JS)
- **Design**: Netflix-inspired dark theme
- **Components**:
  - User input form
  - Three recommendation sections
  - Movie cards with hover effects
- **Features**: Responsive layout, AJAX calls to backend

### Recommendation Algorithm

1. **Top Picks**: Movies sorted by rating count and average rating
2. **Because You Watched**: Genre-based recommendations using cosine similarity
3. **Users Like You**: Collaborative filtering based on similar user ratings

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sriharshith-2006/AIKR.git
   cd AIKR
   ```

2. **Install Python dependencies**
   ```bash
   pip install flask flask-cors pandas scikit-learn numpy
   ```

3. **Run the Flask server**
   ```bash
   python movie.py
   ```
   Server starts on `http://127.0.0.1:5000`

4. **Open the web interface**
   Open `index.html` in your browser or serve it with a web server.

### Usage

1. Enter a username (e.g., "sri", "alex", "john") or user ID (1-610)
2. Click "Get recommendations"
3. View three sections:
   - 🔥 Top Picks for You
   - 🎯 Because you watched [Movie]
   - 👥 Users like you also watched

## 📁 Project Structure

```
AIKR PROJECT/
├── app.py                 # Streamlit app (alternative interface)
├── movie.py               # Flask backend API
├── index.html             # Main web page
├── index.js               # Frontend JavaScript
├── movies.csv             # Movie dataset
├── ratings.csv            # User ratings
├── tags.csv               # User tags
├── links.csv              # External links
├── final_dataset.csv      # Processed dataset
├── Implementation.ipynb   # Jupyter notebook
├── movie.ipynb            # Additional notebook
├── README.txt             # Dataset documentation
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🔧 API Documentation

### GET /recommend/<username>

Returns personalized movie recommendations.

**Parameters:**
- `username`: String or integer user identifier

**Response:**
```json
{
  "top_picks": ["Movie Title 1", "Movie Title 2", ...],
  "because": {
    "movie": "Watched Movie Title",
    "recommendations": ["Rec 1", "Rec 2", ...]
  },
  "users_like_you": ["Similar User Movie 1", ...]
}
```

## 🎨 Web Interface Design

### Theme
- **Background**: Dark (#141414)
- **Accent Color**: Red (#e50914)
- **Typography**: Arial sans-serif
- **Layout**: Card-based with horizontal scrolling

### Sections
1. **Header**: Title and input form
2. **Top Picks**: Horizontal scroll of popular movies
3. **Because You Watched**: Recommendations based on watched movies
4. **Users Like You**: Collaborative recommendations

### Responsive Features
- Mobile-friendly card layout
- Hover effects on movie cards
- Smooth scrolling sections

## 🔍 Algorithm Details

### Content-Based Filtering
- **Input**: Movie genres as text
- **Processing**: TF-IDF vectorization
- **Similarity**: Cosine similarity matrix
- **Output**: Top 5 similar movies per watched film

### Collaborative Filtering
- **User-Item Matrix**: Pivot table of ratings
- **Similarity**: Cosine similarity between users
- **Recommendations**: Movies highly rated by similar users

### Popularity-Based
- **Scoring**: Combination of rating count and average rating
- **Ranking**: Top 10 movies by popularity

## 📈 Future Enhancements

### Backend Improvements
- [ ] Add user authentication
- [ ] Implement rating submission
- [ ] Add movie search functionality
- [ ] Include tags in recommendations
- [ ] Add timestamp-based recency

### Frontend Enhancements
- [ ] Add movie posters from TMDB API
- [ ] Implement user profiles
- [ ] Add rating stars display
- [ ] Create genre filter buttons
- [ ] Add loading animations

### Algorithm Upgrades
- [ ] Hybrid recommendation system
- [ ] Neural collaborative filtering
- [ ] Content-based with metadata
- [ ] A/B testing framework

### Deployment
- [ ] Docker containerization
- [ ] Cloud deployment (Heroku/AWS)
- [ ] Database integration (PostgreSQL)
- [ ] API rate limiting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project uses the MovieLens dataset under GroupLens research license. See `README.txt` for details.

## 🙏 Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **MovieLens** for the recommendation inspiration
- **Scikit-learn** for machine learning tools
- **Flask** for the web framework

## 📞 Contact

**Sriharshith**
- GitHub: [@sriharshith-2006](https://github.com/sriharshith-2006)
- Project Link: [https://github.com/sriharshith-2006/AIKR](https://github.com/sriharshith-2006/AIKR)

---

*Built with ❤️ for movie lovers everywhere*