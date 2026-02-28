"""
build_features.py — Feature engineering and NER pipeline for Paper Trail.

Steps:
  1. Load raw corpus from data/raw/raw_corpus.jsonl
  2. Run NER (spaCy or HuggingFace) to extract PERSON, ORG, DATE, GPE entities
  3. Link detected persons to consequence labels
  4. Generate TF-IDF vectors and sentence embeddings
  5. Save all processed features to data/processed/
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_CORPUS = Path("data/raw/raw_corpus.jsonl")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CONSEQUENCE_LABELS_PATH = PROCESSED_DIR / "consequence_labels.json"
FEATURES_PATH = PROCESSED_DIR / "features.json"
TFIDF_PATH = PROCESSED_DIR / "tfidf_matrix.pkl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"

# ---------------------------------------------------------------------------
# Consequence label lookup
# ---------------------------------------------------------------------------

CONSEQUENCE_LABELS: dict[str, int] = {
    # Tier 0 — Charged / Convicted
    "Ghislaine Maxwell": 0,
    "Jeffrey Epstein": 0,
    "Jean-Luc Brunel": 0,
    # Tier 1 — Settled Civilly
    "Prince Andrew": 1,
    "Virginia Giuffre": 1,
    # Tier 2 — Named / Investigated Only
    "Alan Dershowitz": 2,
    "Alexander Acosta": 2,
    "Bill Richardson": 2,
    "George Mitchell": 2,
    "Les Wexner": 2,
    "Leslie Wexner": 2,
    "Jes Staley": 2,
    "Glenn Dubin": 2,
    "Tom Pritzker": 2,
    "Leon Black": 2,
    "Ehud Barak": 2,
    "Bill Gates": 2,
    "Bill Clinton": 2,
    "Donald Trump": 2,
    "Kevin Spacey": 2,
    "Chris Tucker": 2,
    "Naomi Campbell": 2,
    "Larry Summers": 2,
    "Steven Pinker": 2,
    "Marvin Minsky": 2,
    "Joi Ito": 2,
    "David Copperfield": 2,
    # Tier 3 — No Consequences
    "Al Gore": 3,
    "Paris Hilton": 3,
    "Courtney Love": 3,
    "Woody Allen": 3,
    "Steve Bannon": 3,
}

TIER_NAMES = {
    0: "Charged/Convicted",
    1: "Settled Civilly",
    2: "Named/Investigated Only",
    3: "No Consequences",
}


def save_consequence_labels() -> None:
    """Persist the consequence label lookup table to disk."""
    payload = {
        "labels": CONSEQUENCE_LABELS,
        "tier_names": {str(k): v for k, v in TIER_NAMES.items()},
    }
    CONSEQUENCE_LABELS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    log.info("Consequence labels saved → %s", CONSEQUENCE_LABELS_PATH)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus(path: Path = RAW_CORPUS) -> list[dict]:
    """Load the raw JSONL corpus into a list of record dicts.

    Args:
        path: Path to raw_corpus.jsonl.

    Returns:
        List of record dicts.
    """
    if not path.exists():
        log.error("Corpus not found at %s. Run make_dataset.py first.", path)
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# NER pipeline
# ---------------------------------------------------------------------------


def load_ner_model():
    """Load spaCy NER model (en_core_web_trf preferred, falls back to sm).

    Returns:
        A loaded spaCy Language object.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_trf")
            log.info("NER: loaded en_core_web_trf")
        except OSError:
            log.warning("en_core_web_trf not found, falling back to en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception as exc:
        log.error("Could not load spaCy model: %s", exc)
        return None


def extract_entities(text: str, nlp) -> dict[str, list[str]]:
    """Run NER on text and return entities grouped by label.

    Args:
        text: Input text to process.
        nlp: Loaded spaCy model.

    Returns:
        Dict mapping entity labels (PERSON, ORG, DATE, GPE) to lists of strings.
    """
    doc = nlp(text[:100_000])  # spaCy limit guard
    entities: dict[str, list[str]] = {"PERSON": [], "ORG": [], "DATE": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text.strip())
    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}


def run_ner_pipeline(records: list[dict]) -> list[dict]:
    """Run NER on all corpus records and attach entity annotations.

    Args:
        records: List of corpus record dicts (must have 'text' key).

    Returns:
        Same list with 'entities' and 'named_persons' keys added to each record.
    """
    nlp = load_ner_model()
    if nlp is None:
        log.warning("NER skipped — no spaCy model available.")
        return records

    annotated = []
    for i, record in enumerate(records):
        if i % 50 == 0:
            log.info("NER: processing record %d/%d", i, len(records))
        text = record.get("text", "")
        entities = extract_entities(text, nlp)
        record = dict(record)  # shallow copy
        record["entities"] = entities

        # Link to consequence labels
        named_persons = {}
        for person in entities.get("PERSON", []):
            for key, tier in CONSEQUENCE_LABELS.items():
                if key.lower() in person.lower() or person.lower() in key.lower():
                    named_persons[person] = {"tier": tier, "label": TIER_NAMES[tier]}
                    break
        record["named_persons"] = named_persons
        annotated.append(record)

    return annotated


# ---------------------------------------------------------------------------
# TF-IDF features
# ---------------------------------------------------------------------------


def build_tfidf(texts: list[str], max_features: int = 5000):
    """Fit a TF-IDF vectorizer on a list of texts and return matrix + vectorizer.

    Args:
        texts: List of input texts.
        max_features: Maximum vocabulary size.

    Returns:
        Tuple of (sparse TF-IDF matrix, fitted TfidfVectorizer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    log.info("TF-IDF matrix: %s", matrix.shape)
    return matrix, vectorizer


# ---------------------------------------------------------------------------
# Sentence embeddings
# ---------------------------------------------------------------------------


def build_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                     batch_size: int = 32) -> np.ndarray:
    """Generate sentence embeddings using sentence-transformers.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace sentence-transformers model name.
        batch_size: Encoding batch size.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    try:
        from sentence_transformers import SentenceTransformer

        log.info("Loading sentence transformer: %s", model_name)
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        log.info("Embeddings shape: %s", embeddings.shape)
        return embeddings
    except Exception as exc:
        log.error("Sentence embeddings failed: %s", exc)
        return np.zeros((len(texts), 384))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    log.info("=== Paper Trail — Feature Engineering ===")

    save_consequence_labels()

    records = load_corpus()
    if not records:
        log.error("No records to process. Exiting.")
        raise SystemExit(1)

    annotated = run_ner_pipeline(records)

    texts = [r.get("text", "") for r in annotated]

    # TF-IDF
    log.info("Building TF-IDF features...")
    tfidf_matrix, vectorizer = build_tfidf(texts)
    with TFIDF_PATH.open("wb") as f:
        pickle.dump({"matrix": tfidf_matrix, "vectorizer": vectorizer}, f)
    log.info("TF-IDF saved → %s", TFIDF_PATH)

    # Sentence embeddings
    log.info("Building sentence embeddings...")
    embeddings = build_embeddings(texts)
    np.save(str(EMBEDDINGS_PATH), embeddings)
    log.info("Embeddings saved → %s", EMBEDDINGS_PATH)

    # Save annotated features
    with FEATURES_PATH.open("w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
    log.info("Features saved → %s", FEATURES_PATH)

    log.info("Done.")
