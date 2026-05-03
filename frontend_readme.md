# AIKR Frontend Documentation

This document provides an overview of the frontend architecture, technologies used, and how the user interface integrates with the Flask backend for the AI Movie Recommender (AIKR) project.

## 🛠️ Tech Stack

The frontend is built entirely without heavy frameworks to ensure maximum performance, simplicity, and ease of understanding.

*   **HTML5**: Provides the structural foundation of the application.
*   **CSS3**: Handles all the premium styling, layout, and animations.
    *   **Modern CSS Features**: Uses CSS Variables (`:root`) for consistent theming, Flexbox for responsive row and column layouts, and `backdrop-filter` for glassmorphism effects on the navigation bar.
    *   **Micro-interactions**: Uses CSS `transition` and `transform` properties to create smooth, premium hover effects on movie cards and buttons.
    *   **Custom Scrollbars**: Implements hidden horizontal scrollbars for a clean, app-like carousel experience.
*   **Vanilla JavaScript (ES6+)**: Handles all interactivity and API communication.
    *   **Fetch API**: Used for making asynchronous HTTP requests to the backend.
    *   **DOM Manipulation**: Dynamically updates the HTML content based on the data received.
*   **Typography**: Uses the **Inter** font family from Google Fonts for clean, highly legible text.

## 🔗 How the Frontend Links to the Backend

The frontend acts as a dynamic consumer of the Flask REST API (`movie.py`). Here is the step-by-step flow of how the connection works:

### 1. The Trigger
The navigation bar contains an input field and a "Login" button. 
*   When the page loads, or when the user clicks the "Login" button, the `login()` JavaScript function is executed.
*   The function grabs the username entered in the input field (defaults to `sri` if empty).

### 2. The API Call
The frontend uses the modern JavaScript `fetch()` method to send a `GET` request to the Flask server:
```javascript
fetch("http://127.0.0.1:5000/recommend/" + user)
```
This requests the recommendation data specific to that user.

### 3. Processing the Response
Once the Flask server responds with the JSON data, the frontend intercepts it. The expected JSON structure is:
```json
{
  "top_picks": ["Movie 1", "Movie 2", ...],
  "because": [
    {
      "because": "Watched Movie",
      "recommendations": ["Rec 1", "Rec 2"]
    }
  ]
}
```

### 4. Dynamic DOM Injection
The JavaScript then dynamically generates HTML markup based on this data and injects it into specific "container" elements in the UI:

*   **Top Picks**: It maps over the `top_picks` array, generates a movie card for each title, and injects the resulting HTML string into the `<div id="top-picks-container">`.
*   **Because You Watched**: It iterates over the `because` array. For every movie the user has watched, it creates a new subsection heading and a horizontal carousel of recommended movie cards. This is injected into the `<div id="because-container">`.

### 5. Visual Fallbacks
Because we do not have actual movie poster URLs in the dataset, the frontend uses a helper function (`getGradientClass(index)`) to automatically assign distinct, premium CSS linear gradients to the movie cards so the UI remains visually appealing. It also randomly generates visual "Match Percentages" and progress bars to simulate a complete production-ready streaming interface.

## 🚀 How to Run the Project

You need to run the Flask backend server so the frontend has data to fetch, and then simply open the HTML file!

### Step 1: Start the Backend (Flask API)
The frontend expects data from your Flask server, not the Streamlit one. Open a terminal and run:
```bash
cd AIKR
python movie.py
```
*You should see output indicating the Flask server is running on `http://127.0.0.1:5000`.*

### Step 2: Open the Frontend
Since the frontend is pure HTML/CSS/JS, you have two options:
**Option A:** Simply double-click the `index.html` file inside the `AIKR` folder to open it directly in your web browser.
**Option B:** If you prefer running a local server for the frontend, open a new terminal window:
```bash
cd AIKR
python -m http.server 8000
```
Then visit `http://localhost:8000` in your web browser.

Once open, type `sri`, `alex`, or `john` into the top right input box and click "Login" to see the dynamic recommendations!
