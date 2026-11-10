"""Train the three-class classifier on HateXplain's published data splits."""

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    from ml_pipeline.preprocessor import TextPreprocessor
    from ml_pipeline.config import HATE_OVERRIDE_THRESHOLD, BERT_ENSEMBLE_WEIGHT
except ImportError:
    from preprocessor import TextPreprocessor
    from config import HATE_OVERRIDE_THRESHOLD, BERT_ENSEMBLE_WEIGHT


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "training_data" / "hatexplain.json"
SPLITS_PATH = ROOT / "training_data" / "hatexplain_splits.json"
DYNA_PATH = ROOT / "training_data" / "dynahate.csv"
CLASS_MAP = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral Language"}
LABEL_MAP = {"hatespeech": 0, "offensive": 1, "normal": 2}


def load_hatexplain():
    if not DATASET_PATH.exists() or not SPLITS_PATH.exists():
        raise FileNotFoundError("HateXplain data is required for training. See training_data/README.md.")
    with DATASET_PATH.open(encoding="utf-8") as handle:
        posts = json.load(handle)
    with SPLITS_PATH.open(encoding="utf-8") as handle:
        splits = json.load(handle)

    preprocessor = TextPreprocessor()
    result = {}
    for split in ("train", "val", "test"):
        texts, labels = [], []
        for post_id in splits[split]:
            post = posts.get(post_id)
            if not post:
                continue
            votes = Counter(a["label"] for a in post["annotators"])
            label, count = votes.most_common(1)[0]
            if count < 2 or label not in LABEL_MAP:
                continue
            text = preprocessor.clean_text(" ".join(post["post_tokens"]))
            if text:
                texts.append(text)
                labels.append(LABEL_MAP[label])
        result[split] = (texts, labels)
    return result


def make_vectorizer():
    # Character n-grams help with spelling variants; word n-grams retain context.
    # Keep stopwords and negation, which matter to short sentences.
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=70000,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=100000, sublinear_tf=True, strip_accents="unicode")),
    ])


def make_logistic_regression():
    return LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)


def make_decision_tree():
    return DecisionTreeClassifier(max_depth=25, min_samples_leaf=10,
                                  class_weight="balanced", random_state=42)


def load_dynahate():
    if not DYNA_PATH.exists():
        raise FileNotFoundError("DynaHate data is required for training. See training_data/README.md.")
    frame = pd.read_csv(DYNA_PATH)
    preprocessor = TextPreprocessor()
    frame = frame[frame["split"].isin(["train", "dev", "test"])].copy()
    frame["text"] = frame["text"].fillna("").map(preprocessor.clean_text)
    frame = frame[frame["text"].str.len() > 0]
    frame["target"] = (frame["label"] == "hate").astype(int)
    return {split: (part["text"].tolist(), part["target"].tolist())
            for split, part in frame.groupby("split")}


def make_binary_hate_model():
    return LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)


def combine_predictions(base_probabilities, hate_risk):
    predictions = np.argmax(base_probabilities, axis=1)
    return np.where((predictions != 0) & (hate_risk >= HATE_OVERRIDE_THRESHOLD), 0, predictions)


def evaluate(model, vectors, labels):
    return evaluate_predictions(labels, model.predict(vectors))


def evaluate_predictions(labels, predictions):
    return {
        "accuracy": round(100 * accuracy_score(labels, predictions), 2),
        "report": classification_report(labels, predictions, labels=[0, 1, 2],
                                        target_names=list(CLASS_MAP.values()),
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }


def train_and_evaluate(output_dir=None):
    try:
        from ml_pipeline.transformer import TransformerClassifier
    except ImportError:
        from transformer import TransformerClassifier
    output_dir = Path(output_dir) if output_dir else ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_hatexplain()
    dyna_splits = load_dynahate()
    train_texts, train_labels = splits["train"]
    val_texts, val_labels = splits["val"]
    test_texts, test_labels = splits["test"]

    # The published test posts are never fitted into the vocabulary or model.
    vectorizer = make_vectorizer()
    train_vectors = vectorizer.fit_transform(train_texts)
    val_vectors = vectorizer.transform(val_texts)
    lr = make_logistic_regression().fit(train_vectors, train_labels)
    dt = make_decision_tree().fit(train_vectors, train_labels)
    validation = {
        "logistic_regression": evaluate(lr, val_vectors, val_labels),
        "decision_tree": evaluate(dt, val_vectors, val_labels),
    }

    hate_vectorizer = make_vectorizer()
    dyna_train_texts, dyna_train_labels = dyna_splits["train"]
    hate_train_vectors = hate_vectorizer.fit_transform(dyna_train_texts)
    hate_model = make_binary_hate_model().fit(hate_train_vectors, dyna_train_labels)
    validation["combined"] = evaluate_predictions(
        val_labels,
        combine_predictions(lr.predict_proba(val_vectors),
                            hate_model.predict_proba(hate_vectorizer.transform(val_texts))[:, 1]),
    )

    final_texts = train_texts + val_texts
    final_labels = train_labels + val_labels
    final_vectorizer = make_vectorizer()
    final_vectors = final_vectorizer.fit_transform(final_texts)
    final_lr = make_logistic_regression().fit(final_vectors, final_labels)
    final_dt = make_decision_tree().fit(final_vectors, final_labels)
    dyna_dev_texts, dyna_dev_labels = dyna_splits["dev"]
    final_hate_vectorizer = make_vectorizer()
    final_hate_vectors = final_hate_vectorizer.fit_transform(dyna_train_texts + dyna_dev_texts)
    final_hate_model = make_binary_hate_model().fit(
        final_hate_vectors, dyna_train_labels + dyna_dev_labels
    )
    test_vectors = final_vectorizer.transform(test_texts)
    test_results = {
        "logistic_regression": evaluate(final_lr, test_vectors, test_labels),
        "decision_tree": evaluate(final_dt, test_vectors, test_labels),
    }
    test_results["combined"] = evaluate_predictions(
        test_labels,
        combine_predictions(final_lr.predict_proba(test_vectors),
                            final_hate_model.predict_proba(final_hate_vectorizer.transform(test_texts))[:, 1]),
    )
    transformer = TransformerClassifier(output_dir / "bert_tiny")
    bert_val_probs = transformer.predict_proba(val_texts)
    bert_test_probs = transformer.predict_proba(test_texts)
    test_results["transformer"] = evaluate_predictions(test_labels, np.argmax(bert_test_probs, axis=1))
    ensemble_test_probs = ((1 - BERT_ENSEMBLE_WEIGHT) * final_lr.predict_proba(test_vectors)
                           + BERT_ENSEMBLE_WEIGHT * bert_test_probs)
    test_results["ensemble"] = evaluate_predictions(test_labels, np.argmax(ensemble_test_probs, axis=1))
    test_results["ensemble_with_hate_review"] = evaluate_predictions(
        test_labels,
        combine_predictions(ensemble_test_probs,
                            final_hate_model.predict_proba(final_hate_vectorizer.transform(test_texts))[:, 1]),
    )
    validation["transformer"] = evaluate_predictions(val_labels, np.argmax(bert_val_probs, axis=1))
    validation_ensemble = ((1 - BERT_ENSEMBLE_WEIGHT) * lr.predict_proba(val_vectors)
                           + BERT_ENSEMBLE_WEIGHT * bert_val_probs)
    validation["ensemble"] = evaluate_predictions(val_labels, np.argmax(validation_ensemble, axis=1))
    validation["ensemble_with_hate_review"] = evaluate_predictions(
        val_labels,
        combine_predictions(validation_ensemble,
                            hate_model.predict_proba(hate_vectorizer.transform(val_texts))[:, 1]),
    )

    joblib.dump(final_vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(final_lr, output_dir / "classifier.joblib")
    joblib.dump(final_dt, output_dir / "decision_tree.joblib")
    joblib.dump(final_hate_vectorizer, output_dir / "hate_vectorizer.joblib")
    joblib.dump(final_hate_model, output_dir / "hate_classifier.joblib")
    counts = Counter(final_labels)
    metrics = {
        "dataset_info": {
            "source": "HateXplain (Mathew et al., AAAI 2021)",
            "source_url": "https://github.com/hate-alert/HateXplain",
            "additional_source": "DynaHate (Vidgen et al., ACL 2021)",
            "additional_source_url": "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset",
            "total_samples": len(final_labels),
            "test_samples": len(test_labels),
            "classes": CLASS_MAP,
            "class_counts": {CLASS_MAP[k]: counts[k] for k in CLASS_MAP},
            "vocabulary_size": sum(len(part.vocabulary_) for _, part in final_vectorizer.transformer_list),
        },
        "logistic_regression": {"name": "TF-IDF Logistic Regression", **test_results["logistic_regression"]},
        "transformer": {"name": "Fine-tuned BERT-tiny", **test_results["transformer"]},
        "ensemble": {"name": "TF-IDF and BERT-tiny ensemble",
                     "bert_weight": BERT_ENSEMBLE_WEIGHT, **test_results["ensemble"]},
        "primary_model": {"name": "TF-IDF and BERT-tiny with binary hate review",
                          "bert_weight": BERT_ENSEMBLE_WEIGHT,
                          "hate_override_threshold": HATE_OVERRIDE_THRESHOLD,
                          **test_results["ensemble_with_hate_review"]},
        "decision_tree": {"name": "TF-IDF Decision Tree", **test_results["decision_tree"]},
        "validation": validation,
        "evaluation_note": "Accuracy is measured on HateXplain's held-out test posts, not arbitrary new text.",
        "preprocessing_pipeline": ["Lowercase and remove URLs/mentions", "Preserve stopwords and negation",
                                   "Word 1-2 grams and character 3-5 grams", "TF-IDF weighting",
                                   "BERT-tiny wordpiece context model"],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Training/validation: {len(final_labels)}; held-out test: {len(test_labels)}")
    print(f"Logistic Regression test accuracy: {test_results['logistic_regression']['accuracy']}%")
    print(f"Combined test accuracy: {test_results['combined']['accuracy']}%")
    print(f"BERT-tiny test accuracy: {test_results['transformer']['accuracy']}%")
    print(f"Ensemble test accuracy: {test_results['ensemble']['accuracy']}%")
    print(f"Primary test accuracy: {test_results['ensemble_with_hate_review']['accuracy']}%")
    print(f"Decision Tree test accuracy: {test_results['decision_tree']['accuracy']}%")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
