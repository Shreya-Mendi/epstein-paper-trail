"""
ner.py — NER pipeline wrapper for Paper Trail API.

Provides a thin interface over the spaCy NER model used in build_features.py,
suitable for on-demand inference in the FastAPI backend.
"""

import logging
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

ENTITY_LABELS = {"PERSON", "ORG", "DATE", "GPE"}


@lru_cache(maxsize=1)
def _load_nlp():
    """Load and cache the spaCy NER model.

    Returns:
        Loaded spaCy Language object, or None if unavailable.
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
        log.error("Failed to load spaCy model: %s", exc)
        return None


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text.

    Args:
        text: Raw input text.

    Returns:
        Dict mapping entity labels (PERSON, ORG, DATE, GPE) to unique entity strings.
    """
    nlp = _load_nlp()
    result: dict[str, list[str]] = {label: [] for label in ENTITY_LABELS}

    if nlp is None:
        return result

    doc = nlp(text[:100_000])  # guard against very long inputs
    seen: dict[str, set] = {label: set() for label in ENTITY_LABELS}

    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS and ent.text not in seen[ent.label_]:
            result[ent.label_].append(ent.text.strip())
            seen[ent.label_].add(ent.text)

    return result


def extract_persons(text: str) -> list[str]:
    """Extract only PERSON entities from text.

    Args:
        text: Raw input text.

    Returns:
        List of unique person name strings.
    """
    return extract_entities(text).get("PERSON", [])
