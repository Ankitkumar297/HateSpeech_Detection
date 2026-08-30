import os
import sys
from typing import Dict, Any, Optional

try:
    import joblib  # type: ignore
except ImportError:
    joblib = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

# Ensure root directory is on sys.path for direct execution or IDE tools
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from ml_pipeline.preprocessor import TextPreprocessor
except ImportError:
    from preprocessor import TextPreprocessor  # type: ignore

CLASS_NAMES: Dict[int, str] = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral Language"
}

CLASS_SEVERITY: Dict[str, str] = {
    "Hate Speech": "danger",
    "Offensive Language": "warning",
    "Neutral Language": "success"
}

class HateSpeechClassifier:
    """
    Inference engine that integrates TextPreprocessor, CountVectorizer,
    and the trained ML Classifier.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer: Any = None
        self.classifier: Any = None
        self.decision_tree: Any = None
        self.load_models()

    def load_models(self) -> None:
        vec_path = os.path.join(self.models_dir, "vectorizer.joblib")
        clf_path = os.path.join(self.models_dir, "classifier.joblib")
        dt_path = os.path.join(self.models_dir, "decision_tree.joblib")

        if os.path.exists(vec_path) and os.path.exists(clf_path):
            self.vectorizer = joblib.load(vec_path)
            self.classifier = joblib.load(clf_path)
            if os.path.exists(dt_path):
                self.decision_tree = joblib.load(dt_path)
            print("ML Models and CountVectorizer loaded successfully.")
        else:
            print("Models not found on disk. Initializing training...")
            try:
                from ml_pipeline.train import train_and_evaluate
            except ImportError:
                from train import train_and_evaluate  # type: ignore
            train_and_evaluate(output_dir=self.models_dir)
            self.vectorizer = joblib.load(vec_path)
            self.classifier = joblib.load(clf_path)
            if os.path.exists(dt_path):
                self.decision_tree = joblib.load(dt_path)

    def predict(self, text: str) -> dict:
        """
        Runs full inference on user text and returns:
        - prediction label
        - confidence score (%)
        - class probabilities breakdown
        - color code / severity
        - token analysis
        """
        if not text or not text.strip():
            return {
                "error": "Input text cannot be empty."
            }

        cleaned_text = self.preprocessor.clean_text(text)
        stemmed_text = self.preprocessor.tokenize_and_stem(text)

        if not stemmed_text.strip():
            # If all words were filtered out (e.g. only symbols or single generic stopword)
            return {
                "prediction": "Neutral Language",
                "label_id": 2,
                "confidence": 95.0,
                "severity": "success",
                "color": "#10b981",
                "probabilities": {
                    "Hate Speech": 2.5,
                    "Offensive Language": 2.5,
                    "Neutral Language": 95.0
                },
                "clean_text": cleaned_text,
                "stemmed_text": stemmed_text,
                "tokens": self.preprocessor.get_token_details(text)
            }

        # Vectorize
        X_vec = self.vectorizer.transform([stemmed_text])
        
        # Predict with Logistic Regression (with probability calibration)
        probs = self.classifier.predict_proba(X_vec)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100

        pred_label = CLASS_NAMES.get(pred_idx, "Neutral Language")
        severity = CLASS_SEVERITY.get(pred_label, "success")

        # Color mapping: Red for Hate Speech, Orange for Offensive Language, Green for Neutral
        colors = {
            "Hate Speech": "#ef4444",
            "Offensive Language": "#f97316",
            "Neutral Language": "#10b981"
        }

        prob_dict = {
            "Hate Speech": round(float(probs[0]) * 100, 2),
            "Offensive Language": round(float(probs[1]) * 100, 2),
            "Neutral Language": round(float(probs[2]) * 100, 2)
        }

        # Also get Decision tree prediction for comparison if available
        dt_pred = None
        if self.decision_tree:
            dt_idx = int(self.decision_tree.predict(X_vec)[0])
            dt_pred = CLASS_NAMES.get(dt_idx, "Neutral Language")

        return {
            "prediction": pred_label,
            "label_id": pred_idx,
            "confidence": round(confidence, 2),
            "severity": severity,
            "color": colors.get(pred_label, "#10b981"),
            "probabilities": prob_dict,
            "clean_text": cleaned_text,
            "stemmed_text": stemmed_text,
            "tokens": self.preprocessor.get_token_details(text),
            "dt_prediction": dt_pred
        }
