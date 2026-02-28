"""
make_dataset.py — Data collection pipeline for Paper Trail.

Collects public documents from four sources:
  1. Wikipedia pages for key Epstein-case figures
  2. News articles via NewsAPI / Google News RSS
  3. Court documents via CourtListener API
  4. Flight log PDFs via pdfplumber / PyMuPDF

All collected records are merged into data/raw/raw_corpus.jsonl.
"""

import json
import os
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
WIKI_DIR = RAW_DIR / "wikipedia"
NEWS_DIR = RAW_DIR / "news"
COURT_DIR = RAW_DIR / "court_docs"
FLIGHT_DIR = RAW_DIR / "flight_logs"
CORPUS_PATH = RAW_DIR / "raw_corpus.jsonl"

for d in (WIKI_DIR, NEWS_DIR, COURT_DIR, FLIGHT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Key figures
# ---------------------------------------------------------------------------

EPSTEIN_ASSOCIATES = [
    "Jeffrey Epstein",
    "Ghislaine Maxwell",
    "Alexander Acosta",
    "Prince Andrew",
    "Alan Dershowitz",
    "Leslie Wexner",
    "Bill Richardson",
    "George Mitchell",
    "Jean-Luc Brunel",
    "Jes Staley",
    "Glenn Dubin",
    "Tom Pritzker",
    "Leon Black",
    "Ehud Barak",
    "Bill Gates",
    "Donald Trump",
    "Bill Clinton",
    "Kevin Spacey",
    "Chris Tucker",
    "Naomi Campbell",
    "Paris Hilton",
    "Courtney Love",
    "Al Gore",
    "Larry Summers",
    "Steven Pinker",
    "Marvin Minsky",
    "Joi Ito",
    "Woody Allen",
    "Steve Bannon",
    "David Copperfield",
]

# ---------------------------------------------------------------------------
# Source 1: Wikipedia
# ---------------------------------------------------------------------------


def fetch_wikipedia_page(title: str, delay: float = 1.0) -> Optional[dict]:
    """Fetch a Wikipedia page by title using the Wikipedia REST API.

    Args:
        title: The page title to fetch (e.g. "Jeffrey Epstein").
        delay: Seconds to sleep between requests to be polite.

    Returns:
        A dict with keys {id, source, text, date, url, metadata} or None on failure.
    """
    encoded = title.replace(" ", "_")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        r = requests.get(api_url, timeout=15, headers={"User-Agent": "PaperTrail/1.0"})
        r.raise_for_status()
        data = r.json()

        # Also grab full page text via the parse endpoint
        parse_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        pr = requests.get(parse_url, params=params, timeout=15,
                          headers={"User-Agent": "PaperTrail/1.0"})
        pr.raise_for_status()
        pages = pr.json().get("query", {}).get("pages", {})
        full_text = next(iter(pages.values()), {}).get("extract", data.get("extract", ""))

        record = {
            "id": str(uuid.uuid4()),
            "source": "wikipedia",
            "text": full_text,
            "date": None,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "metadata": {
                "title": data.get("title"),
                "description": data.get("description"),
                "categories": [],
            },
        }
        time.sleep(delay)
        return record
    except Exception as exc:
        log.warning("Wikipedia fetch failed for %r: %s", title, exc)
        return None


def collect_wikipedia(figures: list[str]) -> list[dict]:
    """Collect Wikipedia pages for all named figures.

    Args:
        figures: List of person names to look up on Wikipedia.

    Returns:
        List of raw corpus records.
    """
    records = []
    for name in figures:
        log.info("Wikipedia: fetching %r", name)
        record = fetch_wikipedia_page(name)
        if record:
            out_path = WIKI_DIR / f"{name.replace(' ', '_')}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Source 2: News Articles
# ---------------------------------------------------------------------------

NEWS_QUERIES = [
    "Jeffrey Epstein",
    "Epstein plea deal",
    "Epstein victims",
    "Epstein associates charged",
]


def fetch_google_news_rss(query: str, max_items: int = 20) -> list[dict]:
    """Fetch news articles via Google News RSS feed (no API key required).

    Args:
        query: Search query string.
        max_items: Maximum number of articles to return.

    Returns:
        List of raw corpus records.
    """
    encoded_query = requests.utils.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    records = []
    try:
        r = requests.get(rss_url, timeout=15, headers={"User-Agent": "PaperTrail/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        items = soup.find_all("item")[:max_items]
        for item in items:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            link = item.find("link").get_text(strip=True) if item.find("link") else ""
            pub_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else None
            desc = item.find("description").get_text(strip=True) if item.find("description") else ""

            # Attempt to parse date
            date_str = None
            if pub_date:
                try:
                    dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    date_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    date_str = None

            record = {
                "id": str(uuid.uuid4()),
                "source": "news",
                "text": f"{title}\n\n{desc}",
                "date": date_str,
                "url": link,
                "metadata": {"headline": title, "query": query},
            }
            records.append(record)
        time.sleep(1.0)
    except Exception as exc:
        log.warning("News RSS fetch failed for %r: %s", query, exc)
    return records


def collect_news(queries: list[str]) -> list[dict]:
    """Collect news articles for all queries.

    Args:
        queries: List of search queries to run against Google News RSS.

    Returns:
        List of raw corpus records.
    """
    all_records = []
    for query in queries:
        log.info("News: fetching query %r", query)
        records = fetch_google_news_rss(query)
        all_records.extend(records)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped = []
    for r in all_records:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    # Save per-query snapshot
    out_path = NEWS_DIR / "news_articles.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return deduped


# ---------------------------------------------------------------------------
# Source 3: Court Documents — CourtListener API
# ---------------------------------------------------------------------------

COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v3"


def fetch_courtlistener_dockets(search_term: str = "Epstein", jurisdiction: str = "nyed",
                                 max_results: int = 20) -> list[dict]:
    """Search CourtListener for dockets related to a term.

    Args:
        search_term: Keyword to search dockets for.
        jurisdiction: Court jurisdiction code (e.g. "nyed" for S.D.N.Y area).
        max_results: Maximum number of dockets to fetch.

    Returns:
        List of raw corpus records extracted from docket metadata.
    """
    api_key = os.getenv("COURTLISTENER_API_KEY", "")
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}

    params = {
        "q": search_term,
        "court": jurisdiction,
        "page_size": max_results,
        "format": "json",
    }
    records = []
    try:
        r = requests.get(f"{COURTLISTENER_BASE}/dockets/", params=params,
                         headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        for docket in results:
            case_name = docket.get("case_name", "")
            date_filed = docket.get("date_filed")
            docket_number = docket.get("docket_number", "")
            cl_id = docket.get("id")
            url = f"https://www.courtlistener.com/docket/{cl_id}/"

            text = (
                f"Case: {case_name}\n"
                f"Docket: {docket_number}\n"
                f"Court: {docket.get('court', '')}\n"
                f"Date Filed: {date_filed}\n"
                f"Nature of Suit: {docket.get('nature_of_suit', '')}\n"
            )

            record = {
                "id": str(uuid.uuid4()),
                "source": "court",
                "text": text,
                "date": date_filed,
                "url": url,
                "metadata": {
                    "case_name": case_name,
                    "docket_number": docket_number,
                    "court": docket.get("court"),
                },
            }
            records.append(record)

            # Save individual docket
            out_path = COURT_DIR / f"docket_{cl_id}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

        time.sleep(1.5)
    except Exception as exc:
        log.warning("CourtListener fetch failed: %s", exc)

    return records


def collect_court_docs() -> list[dict]:
    """Collect court documents from CourtListener for Epstein-related dockets.

    Returns:
        List of raw corpus records from court documents.
    """
    log.info("Court docs: fetching from CourtListener")
    records = fetch_courtlistener_dockets("Epstein", "nyed", max_results=20)
    records += fetch_courtlistener_dockets("Epstein", "flsd", max_results=20)
    return records


# ---------------------------------------------------------------------------
# Source 4: Flight Log PDFs
# ---------------------------------------------------------------------------


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        log.warning("pdfplumber failed for %s: %s. Trying PyMuPDF...", pdf_path, exc)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
        except Exception as exc2:
            log.error("PyMuPDF also failed for %s: %s", pdf_path, exc2)
    return "\n".join(text_parts)


def collect_flight_logs() -> list[dict]:
    """Collect and parse flight log PDFs from data/raw/flight_logs/.

    Any PDF placed in data/raw/flight_logs/ will be parsed.  In production,
    download the publicly released Epstein flight log PDFs from court exhibits
    or DocumentCloud before running this function.

    Returns:
        List of raw corpus records extracted from flight log PDFs.
    """
    records = []
    pdf_files = list(FLIGHT_DIR.glob("*.pdf"))
    if not pdf_files:
        log.warning(
            "No PDFs found in %s. Place flight log PDFs there before running.", FLIGHT_DIR
        )
        return records

    for pdf_path in pdf_files:
        log.info("Flight logs: parsing %s", pdf_path.name)
        text = extract_pdf_text(pdf_path)
        if not text.strip():
            log.warning("No text extracted from %s", pdf_path.name)
            continue

        record = {
            "id": str(uuid.uuid4()),
            "source": "flight_log",
            "text": text,
            "date": None,
            "url": str(pdf_path),
            "metadata": {"filename": pdf_path.name},
        }

        out_path = FLIGHT_DIR / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Merge into corpus
# ---------------------------------------------------------------------------


def merge_to_corpus(all_records: list[dict], output_path: Path = CORPUS_PATH) -> None:
    """Write all collected records to a single JSONL corpus file.

    Args:
        all_records: List of all raw corpus record dicts.
        output_path: Destination path for the JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Corpus written: %d records → %s", len(all_records), output_path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Paper Trail — Data Collection ===")

    wiki_records = collect_wikipedia(EPSTEIN_ASSOCIATES)
    log.info("Wikipedia: %d records", len(wiki_records))

    news_records = collect_news(NEWS_QUERIES)
    log.info("News: %d records", len(news_records))

    court_records = collect_court_docs()
    log.info("Court docs: %d records", len(court_records))

    flight_records = collect_flight_logs()
    log.info("Flight logs: %d records", len(flight_records))

    all_records = wiki_records + news_records + court_records + flight_records
    merge_to_corpus(all_records)
    log.info("Done. Total records: %d", len(all_records))
