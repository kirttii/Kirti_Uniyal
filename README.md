# RedRob Data & AI Challenge

## Adaptive Semantic Candidate Ranking Engine

This repository contains my solution for the RedRob Data & AI Challenge. The objective of the project is to build an AI-powered candidate ranking system that goes beyond keyword matching and identifies candidates based on the overall relevance of their experience and profile.

## Project Overview

The ranking engine combines semantic search with traditional information retrieval techniques to evaluate candidates across multiple dimensions. Instead of relying only on exact keyword matches, the system considers work experience, technical skills, career progression, company background, behavioral signals, and confidence in the generated ranking.

## Methodology

The ranking process consists of:

- Semantic similarity using Sentence Transformers (`all-MiniLM-L6-v2`)
- BM25-based keyword retrieval
- Hybrid scoring framework incorporating:
  - Career experience
  - Technical skills
  - Company tier
  - Recent experience
  - Behavioral indicators
  - Confidence estimation

Additional components include adaptive similarity thresholds, production evidence detection, career stability analysis, honeypot candidate filtering, and recruiter-oriented reasoning for each ranked profile.

## Technology Stack

- Python
- Sentence Transformers
- PyTorch
- Pandas
- NumPy
- rank-bm25
- tqdm

## Repository Structure

- `candidate_scoring.py` – Main candidate ranking pipeline
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python candidate_scoring.py candidates.jsonl Kirti_Uniyal.csv
```

## Output

The program generates a ranked CSV file containing:

- Rank
- Candidate ID
- Score
- Confidence
- Recruiter Reasoning

## Author

**Kirti Uniyal**

Developed as part of the **RedRob Data & AI Challenge**.
