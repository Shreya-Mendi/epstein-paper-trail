"""
rag.py — Retrieval-Augmented Generation pipeline for Paper Trail.

Uses ChromaDB as the vector store, sentence-transformers for embeddings,
and Claude (Anthropic API) as the generative backbone.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CORPUS_PATH = Path("data/raw/raw_corpus.jsonl")
CHROMA_DIR = Path("chroma_db")

SYSTEM_PROMPT = (
    "You are Paper Trail, an AI that helps the public understand the Epstein case "
    "by analyzing public legal documents. Always cite your sources. Never speculate "
    "beyond what the documents say. Classify individuals by consequence tier when asked.\n\n"
    "Consequence tiers:\n"
    "  0 — Charged/Convicted\n"
    "  1 — Settled Civilly\n"
    "  2 — Named/Investigated Only\n"
    "  3 — No Consequences\n"
)

CHUNK_SIZE = 512   # tokens (approx characters / 4)
CHUNK_OVERLAP = 128


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows of approximately chunk_size tokens.

    Args:
        text: Input text to chunk.
        chunk_size: Approximate token window size (1 token ≈ 4 characters).
        overlap: Overlap between consecutive chunks in tokens.

    Returns:
        List of text chunk strings.
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        start += char_size - char_overlap
    return chunks


def load_corpus_chunks(corpus_path: Path = CORPUS_PATH) -> tuple[list[str], list[dict]]:
    """Load the raw corpus and split each document into overlapping chunks.

    Args:
        corpus_path: Path to raw_corpus.jsonl.

    Returns:
        Tuple of (chunk_texts, chunk_metadatas).
    """
    import json

    chunks, metadatas = [], []
    if not corpus_path.exists():
        log.warning("Corpus not found at %s", corpus_path)
        return chunks, metadatas

    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            meta_extra = record.get("metadata", {}) or {}
            for i, chunk in enumerate(chunk_text(text)):
                chunks.append(chunk)
                metadatas.append({
                    "doc_id": record.get("id", ""),
                    "source": record.get("source", ""),
                    "url": record.get("url", ""),
                    "date": str(record.get("date") or ""),
                    "chunk_index": i,
                    # DOJ PDF fields (empty string if not a doj_pdf record)
                    "efta_id": str(meta_extra.get("efta_id", "")),
                    "dataset": str(meta_extra.get("dataset", "")),
                    "concern_score": str(meta_extra.get("concern_score", "")),
                })

    log.info("Loaded %d chunks from corpus", len(chunks))
    return chunks, metadatas


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline backed by ChromaDB and Claude."""

    def __init__(self, collection_name: str = "paper_trail",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG pipeline.

        Args:
            collection_name: ChromaDB collection name.
            embedding_model: sentence-transformers model for embedding.
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._collection = None
        self._embedder = None
        self._anthropic_client = None

    def _get_embedder(self):
        """Lazy-load the sentence-transformer embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_collection(self):
        """Lazy-load the ChromaDB collection."""
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_anthropic(self):
        """Lazy-load the Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        return self._anthropic_client

    def index_corpus(self, force: bool = False) -> None:
        """Embed and index all corpus chunks into ChromaDB.

        Args:
            force: If True, drop and rebuild the collection even if it exists.
        """
        collection = self._get_collection()
        if collection.count() > 0 and not force:
            log.info("Collection already indexed (%d chunks). Skipping.", collection.count())
            return

        chunks, metadatas = load_corpus_chunks()
        if not chunks:
            log.warning("No chunks to index.")
            return

        embedder = self._get_embedder()
        log.info("Embedding %d chunks...", len(chunks))
        embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=True)

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Upsert in batches of 5000 (ChromaDB limit)
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )
        log.info("Indexed %d chunks into ChromaDB", len(chunks))

    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of dicts with keys: text, source, url, date, score.
        """
        collection = self._get_collection()
        embedder = self._get_embedder()

        q_emb = embedder.encode([query]).tolist()
        # Fetch a larger candidate pool, then select the best mix
        n_candidates = min(top_k * 4, collection.count() or 1)
        results = collection.query(
            query_embeddings=q_emb,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        all_chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            all_chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "date": meta.get("date", ""),
                "score": round(1 - dist, 4),
                "efta_id": meta.get("efta_id", ""),
                "dataset": meta.get("dataset", ""),
                "concern_score": meta.get("concern_score", ""),
            })

        # Prefer diversity: ensure DOJ PDF chunks appear if they exist in candidates
        doj_chunks = [c for c in all_chunks if c["source"] == "doj_pdf"]
        other_chunks = [c for c in all_chunks if c["source"] != "doj_pdf"]

        # Take top DOJ chunks (up to half of top_k) + top other chunks to fill
        n_doj = min(len(doj_chunks), max(1, top_k // 2))
        n_other = top_k - n_doj
        retrieved = doj_chunks[:n_doj] + other_chunks[:n_other]
        # Re-sort by score so Claude gets best context first
        retrieved.sort(key=lambda c: c["score"], reverse=True)
        return retrieved[:top_k]

    def generate(self, query: str, context_chunks: list[dict]) -> dict:
        """Generate a response using Claude with retrieved context.

        Args:
            query: User's question.
            context_chunks: List of retrieved chunk dicts from retrieve().

        Returns:
            Dict with 'answer' (str) and 'sources' (list of citation dicts).
        """
        client = self._get_anthropic()

        context_text = "\n\n---\n\n".join(
            f"[Source {i+1}: {c['source']} | {c['url']}]\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )

        user_message = (
            f"Context documents:\n\n{context_text}\n\n"
            f"---\n\nUser question: {query}\n\n"
            "Answer based only on the context above. Cite sources by their [Source N] label."
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text if response.content else ""
        sources = []
        for i, c in enumerate(context_chunks):
            src: dict = {
                "index": i + 1,
                "source": c["source"],
                "url": c["url"],
                "date": c["date"],
            }
            # Enrich DOJ PDF sources with EFTA document citation
            if c.get("efta_id"):
                src["efta_id"] = c["efta_id"]
                src["dataset"] = c["dataset"]
                # First 120 chars of the retrieved chunk as quote preview
                src["quote"] = c["text"][:120].replace("\n", " ").strip()
            sources.append(src)
        return {"answer": answer, "sources": sources}

    def query(self, question: str, top_k: int = 8) -> dict:
        """End-to-end RAG: retrieve then generate.

        Args:
            question: User's natural language question.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with 'answer' and 'sources'.
        """
        chunks = self.retrieve(question, top_k=top_k)
        return self.generate(question, chunks)


# Singleton instance for use by FastAPI
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Return the singleton RAGPipeline, initializing if needed."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
