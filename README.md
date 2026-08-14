# Paper Trail

🔗 **Live demo:** [shreya-mendi.github.io/epstein-paper-trail/](https://shreya-mendi.github.io/epstein-paper-trail/)

> *"The Documents Don't Lie"*

Paper Trail is an NLP system that classifies legal consequences for individuals named in the Epstein case using entirely public documents. It features a RAG-based chatbot, a named entity recognition (NER) pipeline, consequence classification across four tiers, and a dark interactive timeline UI.

---

## Project Structure

```
paper-trail/
├── scripts/           # Data collection, feature engineering, modeling, evaluation
├── models/            # Saved model artifacts
├── data/              # Raw, processed, and output data
├── notebooks/         # Exploration, modeling, and evaluation notebooks
├── app/
│   ├── backend/       # FastAPI + RAG + NER + classifier
│   └── frontend/      # Dark timeline UI + chatbot
└── .github/           # PR template
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

Copy `.env.example` to `.env` and fill in your API keys:
```
ANTHROPIC_API_KEY=your_key_here
COURTLISTENER_API_KEY=your_key_here   # optional, increases rate limits
NEWS_API_KEY=your_key_here            # newsapi.org free tier
```

---

## Usage

### 1. Collect data
```bash
python scripts/make_dataset.py
```
Fetches from Wikipedia, news APIs, CourtListener, and flight log PDFs. Outputs to `data/raw/raw_corpus.jsonl`.

### 2. Build features
```bash
python scripts/build_features.py
```
Runs NER, generates embeddings, and produces `data/processed/features.json`.

### 3. Train models
```bash
python scripts/model.py
```
Trains all three models (baseline, logistic regression, DistilBERT). Saves artifacts to `models/`.

### 4. Evaluate
```bash
python scripts/evaluate.py
```
Reports F1 scores, RAG Precision@5, hallucination rate, and robustness experiment. Saves plot to `data/outputs/robustness_experiment.png`.

### 5. Run the app
```bash
uvicorn app.backend.main:app --reload
```
Then open `app/frontend/index.html` in your browser (or serve with any static file server).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/people` | All labeled individuals with consequence tiers |
| `POST` | `/chat` | RAG chatbot — accepts `{query: string}` |
| `GET` | `/timeline` | Timeline events sorted by date |
| `GET` | `/person/{name}` | Profile card data for a specific person |

---

## Consequence Tiers

| Tier | Label | Color |
|------|-------|-------|
| 0 | Charged / Convicted | Red |
| 1 | Settled Civilly | Amber |
| 2 | Named / Investigated Only | Yellow |
| 3 | No Consequences | Grey |

---

## Data Sources

- **Wikipedia** — public biographical articles
- **News articles** — NewsAPI / Google News RSS (public)
- **Court documents** — CourtListener API / RECAP PACER archive (public)
- **Flight logs** — publicly released court exhibits / DocumentCloud (public)

---

## Ethics Statement

All data used in this project is entirely public record — documents that have been released by courts, reported by journalists, or published by government bodies. No private information is accessed or stored. The project is purely analytical and educational, aiming to make existing public information more accessible and searchable.

---

## Live Deployment

*Coming soon — placeholder*

---

## License

MIT
