# Hate Speech and Offensive Language Detection System

A machine learning and natural language processing web application built with **Python**, **Flask**, and **Scikit-Learn** to detect and classify toxic text on social media into three distinct categories:

1. **Hate Speech**: Attacks or denigration targeting individuals or groups based on race, religion, ethnicity, gender, sexual orientation, or disability.
2. **Offensive Language**: General profanity, swearing, or vulgar insults that do not target protected identity groups.
3. **Neutral Language**: Regular discussions, positive commentary, academic text, and news.

---

## 📌 Project Overview & Motivation

Online platforms face a massive challenge moderating user-submitted content at scale. A frequent problem in keyword-based moderation is confusing casual slang or vulgar venting with targeted, harmful hate speech.

This project implements a multi-class NLP classification pipeline based on the taxonomy introduced by *Davidson et al. (2017)*:
- Uses **NLTK SnowballStemmer** and custom regex filters to clean text and remove noise.
- Extracts unigram and bigram feature vectors with **CountVectorizer**.
- Compares **Logistic Regression** (with balanced class weights and calibrated probability distributions) against a **Decision Tree Classifier** benchmark.
- Persists user submissions in an **SQLite** database for auditing and dashboard analytics.
- Provides a clean, responsive web interface with live confidence metrics, token explainability, and PDF/CSV export.

---

## 🛠️ Technology Stack

- **Backend**: Python 3, Flask, Flask-CORS
- **Machine Learning & NLP**: Scikit-Learn, NLTK, Pandas, NumPy, Joblib
- **Database**: SQLite3
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+), Chart.js
- **Document Export**: ReportLab (PDF) & CSV

---

## 📂 Directory Structure

```text
HateSpeech_Detection/
├── app.py                      # Flask backend API server and route handlers
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── ml_pipeline/
│   ├── __init__.py
│   ├── preprocessor.py         # Text cleaning, stopwords filtering, SnowballStemmer
│   ├── model.py                # HateSpeechClassifier inference class & probabilities
│   └── train.py                # Dataset definition, model training, and metrics serialization
│
├── database/
│   ├── db.py                   # SQLite connection helpers, schema creation, CRUD queries
│   └── predictions.db          # SQLite database file (created on first run)
│
├── models/
│   ├── vectorizer.joblib       # Serialized CountVectorizer
│   ├── classifier.joblib       # Serialized Logistic Regression model
│   ├── decision_tree.joblib    # Serialized Decision Tree model
│   └── metrics.json            # Accuracy, precision, recall, and confusion matrix data
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern CSS with dark/light themes and responsive layout
│   └── js/
│       └── app.js              # Client logic, Chart.js integrations, API calls, and PDF export
│
└── templates/
    └── index.html              # Main single-page web interface
```

---

## ⚙️ Installation and Setup

### 1. Clone or Download the Repository
```bash
git clone <repo-url>
cd HateSpeech_Detection
```

### 2. Install Required Python Packages
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model
The pre-trained models are already saved in the `models/` directory. If you would like to retrain on the dataset or evaluate metrics:
```bash
python ml_pipeline/train.py
```

### 4. Run the Flask Web Application
```bash
python app.py
```

### 5. Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🔍 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page web app |
| `POST` | `/api/predict` | Accepts `{ "text": "..." }`, returns class, confidence, and token stems |
| `GET` | `/api/stats` | Returns total prediction counts, class breakdown, and 7-day trend |
| `GET` | `/api/admin/predictions` | Paginated moderation records with optional `search` and `category` query filters |
| `DELETE` | `/api/admin/predictions/<id>` | Deletes a logged record by its ID |
| `GET` | `/api/export/csv` | Streams all database records as a CSV download |
| `POST` | `/api/export/pdf` | Generates a downloadable PDF analysis certificate |
| `GET` | `/api/model/info` | Returns model hyperparameters, accuracy, and dataset taxonomy |

---

## 📊 Dataset & Model Notes

- **Preprocessing Pipeline**: Lowercasing ➔ Regex URL & handle removal ➔ Punctuation strip ➔ NLTK stopword filtering ➔ Snowball stemming.
- **N-Grams**: Unigram + Bigram `(1, 2)` feature extraction with `CountVectorizer`.
- **Primary Model**: Logistic Regression with `class_weight='balanced'` and `L-BFGS` solver.
- **Baseline Model**: Decision Tree with `max_depth=25`.

---

## 📜 License
This project is licensed under the MIT License.
