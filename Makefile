.PHONY: install collect features train evaluate serve clean

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

collect:
	python scripts/make_dataset.py

features:
	python scripts/build_features.py

train:
	python scripts/model.py

evaluate:
	python scripts/evaluate.py

serve:
	uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000

pipeline: collect features train evaluate

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf chroma_db/
