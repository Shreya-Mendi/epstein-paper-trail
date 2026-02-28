"""
model.py — Train and predict with three consequence-tier classifiers.

Model 1: Naive Baseline (majority-class DummyClassifier)
Model 2: Classical ML (TF-IDF + Logistic Regression with grid search)
Model 3: Deep Learning (fine-tuned DistilBERT for sequence classification)

All models predict a consequence tier (0–3) from a text excerpt.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("data/outputs")
MODELS_DIR = Path("models")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = PROCESSED_DIR / "features.json"
LABELS_PATH = PROCESSED_DIR / "consequence_labels.json"
TFIDF_PKL = PROCESSED_DIR / "tfidf_matrix.pkl"

BASELINE_OUT = OUTPUTS_DIR / "baseline_predictions.json"
LOGREG_MODEL_PATH = MODELS_DIR / "classical" / "logreg_model.pkl"
DL_MODEL_PATH = MODELS_DIR / "deep_learning"

LOGREG_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
DL_MODEL_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_labeled_dataset(features_path: Path = FEATURES_PATH,
                          labels_path: Path = LABELS_PATH) -> tuple[list[str], list[int]]:
    """Load texts and integer consequence labels from processed features.

    Only records where at least one named person can be matched to a label
    are included.  The label is taken as the *minimum* tier of all persons
    found (i.e. the most severe consequence mentioned in the document).

    Args:
        features_path: Path to features.json produced by build_features.py.
        labels_path: Path to consequence_labels.json.

    Returns:
        Tuple of (texts, labels) where labels are ints in {0, 1, 2, 3}.
    """
    with labels_path.open(encoding="utf-8") as f:
        label_data = json.load(f)
    consequence_labels: dict[str, int] = label_data["labels"]

    with features_path.open(encoding="utf-8") as f:
        records = json.load(f)

    texts, labels = [], []
    for record in records:
        named_persons: dict = record.get("named_persons", {})
        if not named_persons:
            continue
        tiers = [info["tier"] for info in named_persons.values()]
        label = min(tiers)  # most severe consequence
        texts.append(record.get("text", "")[:2048])
        labels.append(label)

    log.info("Labeled dataset: %d samples", len(texts))
    return texts, labels


# ---------------------------------------------------------------------------
# Model 1: Naive Baseline
# ---------------------------------------------------------------------------


def train_baseline(texts: list[str], labels: list[int]) -> dict:
    """Train and evaluate a majority-class DummyClassifier baseline.

    Args:
        texts: List of input text strings (not used for prediction).
        labels: Integer consequence tier labels.

    Returns:
        Dict with predictions and evaluation metrics.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = DummyClassifier(strategy="most_frequent", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test).tolist()

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    log.info("Baseline report:\n%s", classification_report(y_test, y_pred, zero_division=0))

    output = {
        "model": "baseline",
        "predictions": y_pred,
        "ground_truth": y_test,
        "classification_report": report,
    }
    BASELINE_OUT.write_text(json.dumps(output, indent=2))
    log.info("Baseline predictions → %s", BASELINE_OUT)
    return output


# ---------------------------------------------------------------------------
# Model 2: Classical ML
# ---------------------------------------------------------------------------


def train_classical(texts: list[str], labels: list[int]) -> dict:
    """Train a TF-IDF + Logistic Regression classifier with grid search.

    Args:
        texts: List of input text strings.
        labels: Integer consequence tier labels.

    Returns:
        Dict with best model, predictions, and metrics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    log.info("Best C: %s", grid.best_params_)

    y_pred = best_model.predict(X_test).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    log.info("LogReg report:\n%s", classification_report(y_test, y_pred, zero_division=0))

    with LOGREG_MODEL_PATH.open("wb") as f:
        pickle.dump(best_model, f)
    log.info("Logistic Regression model → %s", LOGREG_MODEL_PATH)

    return {
        "model": "logistic_regression",
        "best_params": grid.best_params_,
        "predictions": y_pred,
        "ground_truth": y_test,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Model 3: DistilBERT Fine-tune
# ---------------------------------------------------------------------------


def train_deep_learning(texts: list[str], labels: list[int]) -> dict:
    """Fine-tune DistilBERT for 4-class consequence tier classification.

    Args:
        texts: List of input text strings (truncated to 512 tokens).
        labels: Integer consequence tier labels.

    Returns:
        Dict with predictions and evaluation metrics.
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        import evaluate as hf_evaluate
    except ImportError as exc:
        log.error("Transformers/datasets not available: %s", exc)
        return {"model": "distilbert", "error": str(exc)}

    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 4

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train}).map(tokenize, batched=True)
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    metric = hf_evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=label_ids, average="weighted")

    training_args = TrainingArguments(
        output_dir=str(DL_MODEL_PATH),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(DL_MODEL_PATH / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(DL_MODEL_PATH))
    tokenizer.save_pretrained(str(DL_MODEL_PATH))
    log.info("DistilBERT model saved → %s", DL_MODEL_PATH)

    predictions_output = trainer.predict(test_ds)
    y_pred = np.argmax(predictions_output.predictions, axis=-1).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    log.info("DistilBERT report:\n%s", classification_report(y_test, y_pred, zero_division=0))

    return {
        "model": "distilbert",
        "predictions": y_pred,
        "ground_truth": y_test,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Paper Trail — Model Training ===")

    if not FEATURES_PATH.exists():
        log.error("features.json not found. Run build_features.py first.")
        raise SystemExit(1)

    texts, labels = load_labeled_dataset()

    if len(texts) < 10:
        log.warning(
            "Very few labeled samples (%d). Results may be unreliable. "
            "Collect more data with make_dataset.py.",
            len(texts),
        )

    log.info("Training Model 1: Baseline")
    baseline_results = train_baseline(texts, labels)

    log.info("Training Model 2: Classical ML")
    classical_results = train_classical(texts, labels)

    log.info("Training Model 3: DistilBERT")
    dl_results = train_deep_learning(texts, labels)

    # Save summary
    summary = {
        "baseline": baseline_results.get("classification_report"),
        "logistic_regression": classical_results.get("classification_report"),
        "distilbert": dl_results.get("classification_report"),
    }
    summary_path = OUTPUTS_DIR / "model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Model summary → %s", summary_path)
    log.info("Done.")
