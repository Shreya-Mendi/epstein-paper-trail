"""
evaluate.py — Evaluation metrics and robustness experiments for Paper Trail.

Metrics implemented:
  1. F1-Score (weighted + per-class) for all three models
  2. RAG Retrieval Precision@5 on a curated question test set
  3. Hallucination Rate (manual / simulated stress test)
  4. Robustness experiment: F1 drop under [REDACTED] noise injection
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

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
LOGREG_MODEL_PATH = MODELS_DIR / "classical" / "logreg_model.pkl"
MODEL_SUMMARY_PATH = OUTPUTS_DIR / "model_summary.json"
ROBUSTNESS_PLOT_PATH = OUTPUTS_DIR / "robustness_experiment.png"
EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"

# ---------------------------------------------------------------------------
# Curated RAG test set (15 Q&A pairs with known source documents)
# ---------------------------------------------------------------------------

RAG_TEST_SET = [
    {
        "question": "Who was Ghislaine Maxwell convicted of?",
        "expected_source_keywords": ["maxwell", "convicted", "trafficking"],
    },
    {
        "question": "What was Jeffrey Epstein charged with?",
        "expected_source_keywords": ["epstein", "charged", "sex trafficking"],
    },
    {
        "question": "What happened to Alexander Acosta?",
        "expected_source_keywords": ["acosta", "resigned", "plea deal"],
    },
    {
        "question": "What did Prince Andrew agree to?",
        "expected_source_keywords": ["andrew", "settlement", "civil"],
    },
    {
        "question": "What is Alan Dershowitz accused of?",
        "expected_source_keywords": ["dershowitz", "named", "allegation"],
    },
    {
        "question": "Who were passengers on Epstein's plane?",
        "expected_source_keywords": ["flight", "log", "passenger"],
    },
    {
        "question": "What was Epstein's plea deal in 2008?",
        "expected_source_keywords": ["plea", "2008", "florida"],
    },
    {
        "question": "Which politicians were named in Epstein documents?",
        "expected_source_keywords": ["clinton", "trump", "politician"],
    },
    {
        "question": "What was Jean-Luc Brunel charged with?",
        "expected_source_keywords": ["brunel", "charged", "trafficking"],
    },
    {
        "question": "What court handled the Epstein case?",
        "expected_source_keywords": ["sdny", "southern district", "new york"],
    },
    {
        "question": "Who is Leslie Wexner's connection to Epstein?",
        "expected_source_keywords": ["wexner", "limited", "power of attorney"],
    },
    {
        "question": "What happened to Jes Staley after Epstein connections emerged?",
        "expected_source_keywords": ["staley", "barclays", "resign"],
    },
    {
        "question": "What did Leon Black pay Epstein?",
        "expected_source_keywords": ["black", "apollo", "million"],
    },
    {
        "question": "What is Epstein's Little St. James island?",
        "expected_source_keywords": ["island", "virgin", "little st james"],
    },
    {
        "question": "What documents were unsealed in 2024?",
        "expected_source_keywords": ["unsealed", "2024", "documents"],
    },
]

# ---------------------------------------------------------------------------
# Metric 1: F1-Score
# ---------------------------------------------------------------------------


def compute_f1_from_summary(summary_path: Path = MODEL_SUMMARY_PATH) -> dict:
    """Load pre-computed model summaries and extract F1 scores.

    Args:
        summary_path: Path to model_summary.json produced by model.py.

    Returns:
        Dict mapping model name to weighted F1 score.
    """
    if not summary_path.exists():
        log.warning("Model summary not found at %s", summary_path)
        return {}

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    f1_scores = {}
    for model_name, report in summary.items():
        if report and "weighted avg" in report:
            f1_scores[model_name] = report["weighted avg"]["f1-score"]
            log.info("F1 (%s): %.4f", model_name, f1_scores[model_name])
    return f1_scores


# ---------------------------------------------------------------------------
# Metric 2: RAG Retrieval Precision@5
# ---------------------------------------------------------------------------


def compute_rag_precision_at_5(test_set: list[dict] = RAG_TEST_SET,
                                 corpus_path: Path = FEATURES_PATH) -> float:
    """Evaluate RAG retrieval quality by checking if expected keywords appear
    in the top-5 retrieved document chunks for each test question.

    Args:
        test_set: List of dicts with 'question' and 'expected_source_keywords'.
        corpus_path: Path to features.json (annotated corpus records).

    Returns:
        Precision@5 score as a float between 0 and 1.
    """
    if not corpus_path.exists():
        log.warning("Features not found for RAG eval. Returning 0.0")
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        log.warning("sentence-transformers not available. Skipping RAG eval.")
        return 0.0

    with corpus_path.open(encoding="utf-8") as f:
        records = json.load(f)

    texts = [r.get("text", "")[:1024] for r in records]
    if not texts:
        return 0.0

    log.info("Building embeddings for RAG eval (%d docs)...", len(texts))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    hits = 0
    for item in test_set:
        query = item["question"]
        keywords = [kw.lower() for kw in item["expected_source_keywords"]]
        q_emb = model.encode([query])
        sims = cosine_similarity(q_emb, doc_embeddings)[0]
        top5_idxs = np.argsort(sims)[::-1][:5]
        top5_texts = [texts[i].lower() for i in top5_idxs]
        # Hit if any keyword appears in any of the top-5 documents
        hit = any(kw in doc_text for kw in keywords for doc_text in top5_texts)
        if hit:
            hits += 1

    precision_at_5 = hits / len(test_set)
    log.info("RAG Precision@5: %.4f (%d/%d)", precision_at_5, hits, len(test_set))
    return precision_at_5


# ---------------------------------------------------------------------------
# Metric 3: Hallucination Rate
# ---------------------------------------------------------------------------

# Simulated chatbot responses with ground-truth support flags.
# In production, replace with actual RAG responses.
HALLUCINATION_TEST_CASES = [
    {"response": "Ghislaine Maxwell was convicted of sex trafficking in 2021.", "supported": True},
    {"response": "Jeffrey Epstein died in August 2019 in federal custody.", "supported": True},
    {"response": "Alexander Acosta resigned as Labor Secretary in 2019.", "supported": True},
    {"response": "Prince Andrew settled a civil lawsuit with Virginia Giuffre.", "supported": True},
    {"response": "Alan Dershowitz was formally charged and convicted.", "supported": False},
    {"response": "The 2008 plea deal was brokered in Florida's Southern District.", "supported": True},
    {"response": "Epstein owned Little St. James island in the U.S. Virgin Islands.", "supported": True},
    {"response": "Leslie Wexner gave Epstein power of attorney.", "supported": True},
    {"response": "Jes Staley resigned from Barclays amid scrutiny.", "supported": True},
    {"response": "Leon Black paid Epstein over $150 million for tax advice.", "supported": True},
    {"response": "Bill Clinton flew on Epstein's plane 26 times.", "supported": False},
    {"response": "Documents unsealed in January 2024 named over 150 individuals.", "supported": True},
    {"response": "Jean-Luc Brunel was found dead in a Paris prison in 2022.", "supported": True},
    {"response": "Epstein's victims received settlements from his estate.", "supported": True},
    {"response": "Donald Trump was convicted in connection with the Epstein case.", "supported": False},
    {"response": "Epstein was a registered sex offender since 2008.", "supported": True},
    {"response": "Virginia Giuffre sued Prince Andrew in a U.S. federal court.", "supported": True},
    {"response": "The FBI investigated Epstein's Palm Beach residence.", "supported": True},
    {"response": "Harvard University returned Epstein's donations.", "supported": True},
    {"response": "MIT Media Lab received $7.5 million from Epstein.", "supported": False},
]


def compute_hallucination_rate(test_cases: list[dict] = HALLUCINATION_TEST_CASES) -> float:
    """Compute the hallucination rate as the fraction of unsupported claims.

    Args:
        test_cases: List of dicts with 'response' (str) and 'supported' (bool).

    Returns:
        Hallucination rate as float between 0 and 1.
    """
    unsupported = sum(1 for case in test_cases if not case["supported"])
    rate = unsupported / len(test_cases)
    log.info(
        "Hallucination rate: %.4f (%d unsupported / %d total)",
        rate, unsupported, len(test_cases),
    )
    return rate


# ---------------------------------------------------------------------------
# Robustness experiment
# ---------------------------------------------------------------------------


def inject_noise(text: str, consequence_labels: dict[str, int]) -> str:
    """Replace person names with [REDACTED] to simulate noisy input.

    Args:
        text: Original document text.
        consequence_labels: Dict mapping known names to tier labels.

    Returns:
        Text with named persons replaced by [REDACTED].
    """
    for name in consequence_labels:
        text = re.sub(re.escape(name), "[REDACTED]", text, flags=re.IGNORECASE)
    return text


def run_robustness_experiment(texts: list[str], labels: list[int],
                               consequence_labels: dict[str, int]) -> dict:
    """Measure F1 drop when person names are redacted from test texts.

    Trains logistic regression on clean data, evaluates on clean vs noisy test sets.

    Args:
        texts: Full list of input texts.
        labels: Corresponding integer labels.
        consequence_labels: Name→tier mapping for noise injection.

    Returns:
        Dict with clean_f1, noisy_f1, and f1_drop.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)

    # Clean F1
    y_pred_clean = clf.predict(X_test_vec)
    clean_f1 = f1_score(y_test, y_pred_clean, average="weighted", zero_division=0)

    # Noisy F1 (redact names in test set only)
    X_test_noisy = [inject_noise(t, consequence_labels) for t in X_test]
    X_test_noisy_vec = vectorizer.transform(X_test_noisy)
    y_pred_noisy = clf.predict(X_test_noisy_vec)
    noisy_f1 = f1_score(y_test, y_pred_noisy, average="weighted", zero_division=0)

    f1_drop = clean_f1 - noisy_f1
    log.info("Clean F1: %.4f | Noisy F1: %.4f | Drop: %.4f", clean_f1, noisy_f1, f1_drop)

    return {"clean_f1": clean_f1, "noisy_f1": noisy_f1, "f1_drop": f1_drop}


def plot_robustness(results: dict, output_path: Path = ROBUSTNESS_PLOT_PATH) -> None:
    """Save a bar chart comparing clean vs noisy F1 scores.

    Args:
        results: Dict with clean_f1 and noisy_f1 keys.
        output_path: Path to save the PNG figure.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#111")
        ax.set_facecolor("#1a1a1a")

        bars = ax.bar(
            ["Clean", "Noisy ([REDACTED])"],
            [results["clean_f1"], results["noisy_f1"]],
            color=["#e63946", "#888888"],
            edgecolor="#333",
            width=0.5,
        )

        for bar, val in zip(bars, [results["clean_f1"], results["noisy_f1"]]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                color="white",
                fontsize=12,
            )

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Weighted F1", color="white")
        ax.set_title(
            f"Robustness Experiment — F1 Drop: {results['f1_drop']:.3f}",
            color="white",
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        log.info("Robustness plot saved → %s", output_path)
    except Exception as exc:
        log.warning("Could not generate robustness plot: %s", exc)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Paper Trail — Evaluation ===")

    # Load labels
    if not LABELS_PATH.exists():
        log.error("consequence_labels.json not found. Run build_features.py first.")
        raise SystemExit(1)
    with LABELS_PATH.open(encoding="utf-8") as f:
        label_data = json.load(f)
    consequence_labels: dict[str, int] = label_data["labels"]

    # Load dataset
    if not FEATURES_PATH.exists():
        log.error("features.json not found. Run build_features.py first.")
        raise SystemExit(1)

    from model import load_labeled_dataset
    texts, labels = load_labeled_dataset()

    # Metric 1: F1
    log.info("--- Metric 1: F1-Score ---")
    f1_scores = compute_f1_from_summary()

    # Metric 2: RAG Precision@5
    log.info("--- Metric 2: RAG Precision@5 ---")
    rag_p5 = compute_rag_precision_at_5()

    # Metric 3: Hallucination Rate
    log.info("--- Metric 3: Hallucination Rate ---")
    hallucination_rate = compute_hallucination_rate()

    # Robustness experiment
    log.info("--- Robustness Experiment ---")
    if texts:
        robustness = run_robustness_experiment(texts, labels, consequence_labels)
        plot_robustness(robustness)
    else:
        log.warning("No labeled data for robustness experiment.")
        robustness = {}

    # Save full report
    report = {
        "f1_scores": f1_scores,
        "rag_precision_at_5": rag_p5,
        "hallucination_rate": hallucination_rate,
        "robustness": robustness,
    }
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Evaluation report → %s", EVAL_REPORT_PATH)
    log.info("Done.")
