"""
classifier.py — Consequence tier classifier for Paper Trail API.

Loads the trained logistic regression model (or falls back to the label lookup)
and exposes predict functions for use in the FastAPI backend.

The label lookup now carries richer per-person metadata:
  tier, bio, category, flights, documents, connections,
  in_black_book, nationality, photo_url
"""

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

LOGREG_PATH = Path("models/classical/logreg_model.pkl")
LABELS_PATH = Path("data/processed/consequence_labels.json")

TIER_NAMES = {
    0: "Charged/Convicted",
    1: "Settled Civilly",
    2: "Named/Investigated Only",
    3: "No Consequences",
}

TIER_COLORS = {
    0: "#e63946",
    1: "#f4a261",
    2: "#e9c46a",
    3: "#6c757d",
}


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the trained logistic regression model."""
    if not LOGREG_PATH.exists():
        log.warning("LogReg model not found at %s — using label lookup only.", LOGREG_PATH)
        return None
    with LOGREG_PATH.open("rb") as f:
        model = pickle.load(f)
    log.info("Loaded classifier from %s", LOGREG_PATH)
    return model


@lru_cache(maxsize=1)
def _load_label_lookup() -> dict:
    """Load and cache the consequence label lookup.

    Returns:
        Dict mapping person name -> metadata dict (tier, bio, photo_url, …)
        OR legacy int (for backwards compatibility with old label files).
    """
    if not LABELS_PATH.exists():
        return {}
    with LABELS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("labels", {})


def _person_entry(name: str, raw) -> dict:
    """Normalise a label entry into a full person dict.

    Args:
        name: Person's name.
        raw: Either an int tier (legacy) or a metadata dict.

    Returns:
        Full person dict with name, tier, label, color, and optional extras.
    """
    if isinstance(raw, int):
        tier = raw
        extras = {}
    else:
        tier = raw.get("tier", 2)
        extras = {k: v for k, v in raw.items() if k != "tier"}

    return {
        "name": name,
        "tier": tier,
        "label": TIER_NAMES.get(tier, "Unknown"),
        "color": TIER_COLORS.get(tier, "#888"),
        **extras,
    }


def predict_from_text(text: str) -> dict:
    """Predict consequence tier from a text excerpt.

    Prefers the trained model if available; falls back to tier 2.

    Args:
        text: Document text excerpt.

    Returns:
        Dict with tier, label, color.
    """
    model = _load_model()
    if model is not None:
        try:
            tier = int(model.predict([text])[0])
        except Exception as exc:
            log.warning("Model prediction failed: %s", exc)
            tier = 2
    else:
        tier = 2

    return {
        "tier": tier,
        "label": TIER_NAMES.get(tier, "Unknown"),
        "color": TIER_COLORS.get(tier, "#888"),
    }


def predict_for_person(name: str) -> dict:
    """Look up the known consequence tier and metadata for a named individual.

    Args:
        name: Full name (case-insensitive partial match supported).

    Returns:
        Full person dict.
    """
    lookup = _load_label_lookup()
    raw = lookup.get(name)

    if raw is None:
        name_lower = name.lower()
        for key, val in lookup.items():
            if key.lower() in name_lower or name_lower in key.lower():
                raw = val
                break

    if raw is None:
        raw = 2  # default tier

    return _person_entry(name, raw)


def get_all_labeled_persons() -> list[dict]:
    """Return all persons from the label lookup with full metadata.

    Returns:
        List of person dicts sorted by tier then name.
    """
    lookup = _load_label_lookup()
    persons = [_person_entry(name, raw) for name, raw in lookup.items()]
    persons.sort(key=lambda p: (p["tier"], p["name"]))
    return persons
