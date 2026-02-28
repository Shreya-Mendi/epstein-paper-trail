"""
classifier.py — Consequence tier classifier for Paper Trail API.

Loads the trained logistic regression model (or falls back to the label lookup)
and exposes a predict() function for use in the FastAPI backend.
"""

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

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
    0: "#e63946",   # red
    1: "#f4a261",   # amber
    2: "#e9c46a",   # yellow
    3: "#6c757d",   # grey
}


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the trained logistic regression model.

    Returns:
        Fitted sklearn Pipeline, or None if the model file is missing.
    """
    if not LOGREG_PATH.exists():
        log.warning("LogReg model not found at %s — using label lookup only.", LOGREG_PATH)
        return None
    with LOGREG_PATH.open("rb") as f:
        model = pickle.load(f)
    log.info("Loaded classifier from %s", LOGREG_PATH)
    return model


@lru_cache(maxsize=1)
def _load_label_lookup() -> dict[str, int]:
    """Load and cache the hand-labeled consequence lookup table.

    Returns:
        Dict mapping person name (str) to tier int.
    """
    if not LABELS_PATH.exists():
        return {}
    with LABELS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("labels", {})


def predict_from_text(text: str) -> dict:
    """Predict consequence tier from a text excerpt.

    Prefers the trained model if available; falls back to majority class (tier 2).

    Args:
        text: Document text excerpt.

    Returns:
        Dict with 'tier' (int), 'label' (str), and 'color' (hex str).
    """
    model = _load_model()
    if model is not None:
        try:
            tier = int(model.predict([text])[0])
        except Exception as exc:
            log.warning("Model prediction failed: %s", exc)
            tier = 2
    else:
        tier = 2  # default: Named/Investigated Only

    return {
        "tier": tier,
        "label": TIER_NAMES.get(tier, "Unknown"),
        "color": TIER_COLORS.get(tier, "#888"),
    }


def predict_for_person(name: str) -> dict:
    """Look up the known consequence tier for a named individual.

    Args:
        name: Full name of the person (case-insensitive partial match supported).

    Returns:
        Dict with 'tier', 'label', 'color', and 'name'.
    """
    lookup = _load_label_lookup()
    # Exact match first, then case-insensitive partial match
    tier = lookup.get(name)
    if tier is None:
        name_lower = name.lower()
        for key, t in lookup.items():
            if key.lower() in name_lower or name_lower in key.lower():
                tier = t
                break
    if tier is None:
        tier = 2  # default

    return {
        "name": name,
        "tier": tier,
        "label": TIER_NAMES.get(tier, "Unknown"),
        "color": TIER_COLORS.get(tier, "#888"),
    }


def get_all_labeled_persons() -> list[dict]:
    """Return all persons from the consequence label lookup with their tiers.

    Returns:
        List of dicts with 'name', 'tier', 'label', 'color'.
    """
    lookup = _load_label_lookup()
    return [
        {
            "name": name,
            "tier": tier,
            "label": TIER_NAMES.get(tier, "Unknown"),
            "color": TIER_COLORS.get(tier, "#888"),
        }
        for name, tier in sorted(lookup.items(), key=lambda x: (x[1], x[0]))
    ]
