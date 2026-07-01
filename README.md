# RedRob Data & AI Challenge

## Adaptive Semantic Candidate Ranking Engine

This repository contains my solution for the RedRob Data & AI Challenge.

The goal of the project is to rank candidates intelligently by combining semantic similarity, keyword matching, career history, skills, behavioral signals, company quality, and confidence scoring instead of relying only on keyword search.

## Approach

The ranking pipeline uses:

- Sentence Transformers (all-MiniLM-L6-v2) for semantic matching
- BM25 for keyword relevance
- Hybrid scoring using:
  - Career evidence
  - Skills
  - Company tier
  - Recency
  - Behavioral signals
  - Confidence score

Additional features include adaptive semantic thresholds, production evidence detection, career stability scoring, honeypot detection, and recruiter-friendly reasoning generation.

## Technologies

- Python
- SentenceTransformers
- PyTorch
- Pandas
- NumPy
- rank-bm25
- tqdm

## Repository

- `candidate_scoring_v26.py` – Main ranking pipeline
- `requirements.txt` – Project dependencies
- `README.md` – Documentation

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python candidate_scoring_v26.py candidates.jsonl Kirti_Uniyal.csv
```

## Output

The script generates:

`Kirti_Uniyal.csv`

The output contains:

- Rank
- Candidate ID
- Score
- Confidence
- Recruiter Reasoning

## Author

Kirti Uniyal

Submission for the RedRob Data & AI Challenge.
