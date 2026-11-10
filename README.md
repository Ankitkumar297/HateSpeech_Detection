# Hate Speech and Offensive Language Detection System

A machine learning and natural language processing web application built with **Python**, **Flask**, and **Scikit-Learn** to detect and classify toxic text on social media into three distinct categories:

1. **Hate Speech**: Attacks or denigration targeting individuals or groups based on race, religion, ethnicity, gender, sexual orientation, or disability.
2. **Offensive Language**: General profanity, swearing, or vulgar insults that do not target protected identity groups.
3. **Neutral Language**: Regular discussions, positive commentary, academic text, and news.

---

## 📌 Project Overview & Motivation

Online platforms face a massive challenge moderating user-submitted content at scale. A frequent problem in keyword-based moderation is confusing casual slang or vulgar venting with targeted, harmful hate speech.

This project implements a three-class classifier trained on public labeled data:
- Trains on the published **HateXplain** train/validation splits and evaluates on its held-out test split.
- Uses **DynaHate** as a second, adversarial source for a conservative hate-risk check.
- Keeps negation and common words in the classifier input, combining word and character **TF-IDF** features with a fine-tuned **BERT-tiny** context model.
- Reports measured accuracy for the deployed classifier, the ensemble without hate-risk review, and baselines. Model scores are not calibrated probabilities.
- Persists user submissions in a local **SQLite** database for auditing and dashboard analytics.
- Provides a clean, responsive web interface with live confidence metrics, token explainability, and PDF/CSV export.

---

## 🛠️ Technology Stack

- **Backend**: Python 3, Flask
- **Machine Learning & NLP**: Scikit-Learn, PyTorch, Transformers, NLTK, Pandas, NumPy, Joblib
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
│   ├── preprocessor.py         # Text cleaning and token inspection
│   ├── model.py                # HateSpeechClassifier inference class & probabilities
│   ├── transformer.py          # Fine-tuned BERT-tiny inference
│   ├── train_transformer.py    # Transformer fine-tuning on published splits
│   └── train.py                # TF-IDF training and combined metrics
│
├── database/
│   ├── db.py                   # SQLite connection helpers, schema creation, CRUD queries
│   └── predictions.db          # SQLite database file (created on first run)
│
├── models/
│   ├── vectorizer.joblib       # Serialized three-class TF-IDF vectorizer
│   ├── classifier.joblib       # Serialized Logistic Regression model
│   ├── hate_vectorizer.joblib  # Serialized binary hate-risk vectorizer
│   ├── hate_classifier.joblib  # Serialized binary hate-risk model
│   ├── decision_tree.joblib    # Serialized Decision Tree model
│   ├── bert_tiny/              # Fine-tuned transformer model and tokenizer
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
The trained models are already saved in the `models/` directory. To retrain the transformer and then regenerate the TF-IDF models and combined metrics:
```bash
python -m ml_pipeline.train_transformer
python ml_pipeline/train.py
```
Transformer training writes to `models/bert_tiny/`, uses the published training split, selects an epoch on validation macro-F1, and evaluates the held-out test split once.

### 4. Run the Flask Web Application
```bash
set ADMIN_API_KEY=replace-with-a-long-random-secret
python app.py
```

In PowerShell, set the key with `$env:ADMIN_API_KEY = "replace-with-a-long-random-secret"`. Enter that key in the Logs panel to view, delete, or export records. Set `DATABASE_PATH` to an absolute path on persistent storage if the default local database path is unsuitable. Without `ADMIN_API_KEY`, record management remains locked.

This SQLite setup requires a persistent shared filesystem. Serverless deployments such as Vercel and AWS Lambda are intentionally rejected at startup because their temporary filesystems cannot retain a reliable audit log. Use a persistent database before deploying there.

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
| `POST` | `/api/export/pdf` | Accepts `{ "text": "..." }`, recomputes classification on the server, and generates a PDF report |
| `GET` | `/api/model/info` | Returns model hyperparameters, accuracy, and dataset taxonomy |

Record history, admin actions, and CSV export require an `X-Admin-Key` header matching `ADMIN_API_KEY`.
History pages accept a `limit` of 1–100 and a nonnegative `offset`. Sample audit records are added only when a new local database is created; deleting them will not cause them to return on restart.

---

## 📊 Dataset & Model Notes

- **Preprocessing Pipeline**: Lowercasing ➔ URL and handle removal ➔ punctuation cleanup. Stopwords and negation remain in the classifier input.
- **Features**: Word 1–2 grams and character 3–5 grams with TF-IDF weighting.
- **Primary Model**: 80% three-class Logistic Regression plus 20% fine-tuned BERT-tiny scores, followed by a conservative binary hate-risk check. The ensemble weight was selected on validation macro-F1. No sentence-specific rules are used.
- **Baseline Model**: Decision Tree with `max_depth=25`.

The deployed model scores **66.58%** accuracy on HateXplain's held-out test posts, up from 65.75%. The ensemble without the hate-risk override scores **67.83%**, but misses more direct hate examples. These benchmarks do not guarantee correct results on arbitrary new text; false positives and false negatives remain. Do not use the output as an automatic moderation decision.

---

## 📜 License
This project is licensed under the MIT License.
