import json, gzip
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import re
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
from rank_bm25 import BM25Okapi
warnings.filterwarnings('ignore')
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

TEAM_NAME = "Kirti_Uniyal"
TOP_N = 100
SIM_THRESHOLD_FLOOR = 0.45  # fallback floor for adaptive threshold (replaces unused SIM_THRESHOLD)

def extract_jd_concepts(jd_text):
    return {
        'ranking_system': [
            "Designed and built production recommendation engines that serve millions of users daily",
            "Owned search ranking and relevance systems with learning to rank models",
            "Improved marketplace relevance and candidate matching algorithms in production",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Built learning-to-rank pipelines for production search and recommendation systems",
            "Developed LambdaMART and XGBoost rankers for large-scale ranking problems",
            "Worked on ranking features, re-ranking, and candidate generation for high-traffic products",
            "Shipped production ranking improvements that improved relevance and user engagement",
            "Recommendation engine serving millions of users",
            "Hybrid retrieval and re-ranking system",
            "Ranking candidate matching production"
        ],
        'learning_to_rank': [
            "Built learning-to-rank pipeline with LambdaMART",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Candidate ranking and relevance optimization",
            "Marketplace ranking algorithms"
        ],
        'retrieval_prod': [
            "Implemented hybrid search combining dense retrieval and BM25 at scale",
            "Built vector search infrastructure using approximate nearest neighbors",
            "Designed retrieval pipelines with indexing and embedding drift monitoring",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Built dense, sparse, and hybrid retrieval systems for production search",
            "Implemented semantic search and multi-stage retrieval pipelines for relevance",
            "Worked on embedding-based retrieval and reranking for large corpora",
            "Optimized retrieval quality with BM25, ANN, and learning-to-rank signals"
        ],
        'vector_db_prod': [
            "Production experience with vector databases like Pinecone, Weaviate, and Milvus",
            "Built and scaled FAISS and OpenSearch vector indexing infrastructure",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Managed vector indexing and embedding storage with Pinecone, Weaviate, Milvus, and FAISS",
            "Built ANN search systems with HNSW, IVF, and ScaNN indexing strategies",
            "Worked with pgvector, Qdrant, Chroma, and other vector database backends",
            "Designed vector retrieval pipelines with monitoring for drift and quality"
        ],
        'eval_framework': [
            "Built offline evaluation frameworks using NDCG, MRR, and MAP metrics",
            "Ran A/B tests and experimentation frameworks for ranking systems",
            "Owned retrieval and ranking metrics with online-offline alignment",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Evaluated retrieval quality with NDCG, MRR, MAP, Recall@K, and Precision@K",
            "Built online and offline evaluation frameworks for ranking and search",
            "Ran experimentation and analyzed click-through, engagement, and conversion metrics",
            "Tracked recall, precision, and relevance metrics across model iterations"
        ],
        'production_deploy': [
            "Deployed machine learning models to production serving real users",
            "Built low latency inference systems with high throughput requirements",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Deployed ranking and retrieval models with robust model serving infrastructure",
            "Built low-latency inference pipelines for real-time search and recommendation",
            "Implemented model deployment workflows with monitoring and rollback support",
            "Worked on production ML deployment for ranking, retrieval, and embeddings"
        ],
        'production_scale': [
            "Scaled systems to handle millions of requests per second",
            "Built distributed systems using Spark, Kafka, and Kubernetes",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Designed data pipelines and serving architecture for large-scale search workloads",
            "Built scalable retrieval and ranking systems using Spark, Kafka, and Kubernetes",
            "Handled throughput, latency, and availability requirements for production systems",
            "Scaled ranking and retrieval components across multiple regions and traffic peaks"
        ],
        'production_monitor': [
            "Monitored ML models in production and handled on-call incidents",
            "Maintained SLA uptime 99.9% with latency monitoring and alerting",
            "Built learning-to-rank pipeline with LambdaMART",
            "Recommendation engine serving millions of users",
            "Candidate ranking and relevance optimization",
            "Hybrid retrieval and re-ranking system",
            "Marketplace ranking algorithms",
            "Learning-to-rank with XGBoost",
            "Search relevance optimization",
            "Ranking candidate matching production",
            "Tracked model drift, quality regression, and retrieval performance in production",
            "Built monitoring for latency, throughput, availability, and ranking quality",
            "Owned incident response and mitigation for ranking and retrieval services",
            "Maintained observability dashboards for ML serving and search systems"
        ]
    }

JD_CONCEPTS = extract_jd_concepts("")

CATEGORY_WEIGHTS = {
    'ranking_system': 0.12,
    'eval_framework': 0.09,
    'retrieval_prod': 0.07,
    'vector_db_prod': 0.04,
    'production_deploy': 0.04,
    'production_scale': 0.03,
    'production_monitor': 0.03,
    'learning_to_rank': 0.06
}

SKILL_KEYWORDS = [
    'python', 'java', 'scala', 'go', 'rust', 'c++', 'sql', 'spark', 'kafka', 'airflow',
    'kubernetes', 'docker', 'aws', 'gcp', 'azure', 'pytorch', 'tensorflow', 'xgboost',
    'faiss', 'elasticsearch', 'opensearch', 'pinecone', 'weaviate', 'milvus', 'redis','qdrant',
'pgvector',
'chroma',
'bm25',
'hnsw',
'ann',
'bge',
'langchain',
'llamaindex',
'mlflow',
'kubeflow',
'rag',
'ltr',
'learning to rank',
'sentence-transformers',
'scann',
'ivf',
'qdrant',
'pgvector',
'chroma',
'bm25',
'hnsw',
'ann',
'scann',
'ivf',
'rag',
'learning to rank',
'ltr',
'bge',
'sentence-transformers',
'langchain',
'llamaindex',
'mlflow',
'kubeflow',
'reranker',
'dense retrieval',
'sparse retrieval',
'semantic search',
'vector search',
'embedding generation'
]

# Key technical terms used to sharpen the BM25 corpus (feedback point 5)
JD_KEY_TERMS = [
    'ranking', 'retrieval', 'vector', 'bm25', 'faiss', 'ndcg', 'mrr', 'map',
    'embedding', 'recommendation', 'relevance', 'pinecone', 'milvus', 'weaviate',
    'opensearch', 'elasticsearch', 'latency', 'throughput', 'kafka', 'spark', 'kubernetes','ranking',
'retrieval',
'vector',
'bm25',
'faiss',
'hnsw',
'ann',
'ivf',
'scann',
'ndcg',
'mrr',
'map',
'precision',
'recall',
'embedding',
'relevance',
'dense',
'sparse',
'hybrid',
'rag',
'pinecone',
'milvus',
'weaviate',
'qdrant',
'pgvector',
'chroma',
'lambdaMART',
'xgboost',
'bge',
'mlflow',
'kubeflow',
'dense',
'sparse',
'hybrid',
'recall',
'precision',
'reranker',
'xgboost',
'lambdamart',
'bge',
'ann',
'hnsw',
'ivf',
'scann',
'qdrant',
'pgvector',
'chroma',
'rag',
'mlflow',
'kubeflow',
'learning-to-rank',
'lambdamart',
'xgboost',
're-ranking',
'relevance',
'recommendation',
'candidate ranking',
'marketplace ranking'
]

RISKY_TITLES = ['marketing manager', 'operations manager', 'customer support', 'hr manager', 'accountant', 'civil engineer', 'sales', 'content writer']
PRODUCT_MANAGER_TITLES = ['product manager', 'pm', 'search pm', 'ranking pm', 'ml pm', 'group pm', 'principal pm']
PROMOTION_TITLES = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager', 'director',
                    'sde ii', 'sde 2', 'engineer ii', 'l4', 'l5', 'l6', 'ic3', 'ic4', 'ic5', 'mts']


# FIX #4: more granular 5-tier company scale instead of the previous 3-tier one.
# Tier 5: frontier AI / largest-scale tech. Tier 4: elite high-growth tech.
# Tier 3: large established product companies. Tier 2: solid funded startups.
# Tier 1: everything else (default, unknown, or unrecognized companies).
# This keeps the original company list intact and adds the new companies you requested.
COMPANY_TIERS = {
    5: ['google', 'openai', 'anthropic', 'deepmind', 'meta', 'apple'],
    4: ['amazon', 'microsoft', 'netflix', 'nvidia', 'stripe', 'databricks', 'snowflake'],
    3: ['uber', 'airbnb', 'linkedin', 'salesforce', 'adobe', 'oracle', 'sap', 'flipkart', 'swiggy', 'zomato', 'razorpay'],
    2: ['meesho', 'paytm', 'phonepe', 'ola', 'cred', 'groww', 'nykaa', 'genpact', 'sarvam'],
}

# Keep each company only at its highest tier so lower-tier duplicates do not leak in.
for tier in [5, 4, 3, 2]:
    for company in list(COMPANY_TIERS[tier]):
        for lower_tier in [t for t in [5, 4, 3, 2] if t < tier]:
            if company in COMPANY_TIERS[lower_tier]:
                COMPANY_TIERS[lower_tier].remove(company)


def get_company_tier(company):
    if not company:
        return 1
    comp_lower = company.lower()
    for tier in (5, 4, 3, 2):
        if any(name in comp_lower for name in COMPANY_TIERS[tier]):
            return tier
    return 1


def extract_semantic_evidence(bullet_data, concept_embeddings_map, threshold):
    # FIX #3: previously only the single best-matching category was credited per
    # bullet. A bullet can legitimately describe ranking + retrieval + eval work in
    # one sentence, so now every category above threshold gets credit, not just the
    # top one. This improves recall for multi-skill bullets.
    #
    # FIX #6: `threshold` can be a dict {category: float} so each category uses its
    # own adaptive cutoff (ranking-system similarities and vector-DB similarities
    # tend to have different distributions), or a single float applied uniformly
    # as a fallback.
    career_evidence = {cat: [] for cat in JD_CONCEPTS.keys()}
    if not concept_embeddings_map:
        return career_evidence

    def cat_threshold(cat):
        return threshold.get(cat, 0.6) if isinstance(threshold, dict) else threshold

    for bullet_idx, (emb, text, company, title, months) in enumerate(bullet_data):
        tier = get_company_tier(company)
        for category, concept_embs in concept_embeddings_map.items():
            if concept_embs is None or len(concept_embs) == 0:
                continue
            sims = util.cos_sim(emb, concept_embs)[0]
            max_sim = torch.max(sims).item()
            if max_sim > cat_threshold(category):
                career_evidence[category].append({
                    'quote': text[:80].strip(),
                    'company': company,
                    'title': title,
                    'similarity': max_sim,
                    'company_tier': tier,
                    'duration_months': months,
                    'bullet_idx': bullet_idx
                })

    return career_evidence


def check_negative_evidence(c, career_evidence):
    # FIX #5: stricter check. Previously this could still penalize a candidate who
    # mentions "tutorial" or "course" ANYWHERE in their text, even alongside solid
    # production evidence in a category not in the small whitelist below. Now we
    # check for ANY production-flavored evidence across all categories, and we
    # tighten the regexes to require the negative-context word to appear close to
    # the technical term (within ~6 words) rather than anywhere in the same sentence.
    has_prod_evidence = any(career_evidence[cat] for cat in JD_CONCEPTS.keys())
    if not has_prod_evidence:
        full_text = ' '.join([h.get('description', '') for h in c.get('career_history', [])]) + ' ' + c.get('profile', {}).get('summary', '')
        negative_patterns = [
            r'\b(learning|exploring|interested in|curious about)\b(?:\s+\w+){0,5}\s+\b(rag|llm|vector|ranking)\b(?!\s+to\b)',
            r'\bself[- ]?(taught|learner|directed)\b',
            r'\bside project\b',
            r'\bkaggle\b',
            r'\bcourse\b(?:\s+\w+){0,4}\s+\b(completed|finished|took)\b',
            r'\btutorial\b',
            r'\bbootcamp\b',
            r'\b(personal|hobby|demo|academic|university|capstone)\b(?:\s+\w+){0,4}\s+\b(project|projects)\b(?:\s+\w+){0,4}\s+\b(rag|llm|vector|ranking|retrieval|search|recommendation|embedding)\b',
            r'\b(certification|certifications)\b(?:\s+\w+){0,4}\s+\b(rag|llm|vector|ranking|retrieval|search|recommendation|embedding)\b',
            r'\b(beginner|aspiring|currently learning|experimenting with|trying out|learning journey)\b(?:\s+\w+){0,4}\s+\b(rag|llm|vector|ranking|retrieval|search|recommendation|embedding)\b',
            r'\bside experiments\b(?:\s+\w+){0,4}\s+\b(rag|llm|vector|ranking|retrieval|search|recommendation|embedding)\b'
        ]
        for pattern in negative_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                return True
    return False


def detect_honeypot(c, career_evidence, summary_evidence):
    title = c.get('profile', {}).get('current_title', '').lower()
    is_risky = any(t in title for t in RISKY_TITLES) and not any(p in title for p in PRODUCT_MANAGER_TITLES)

    career_count = sum(len(v) for v in career_evidence.values())
    summary_count = sum(len(v) for v in summary_evidence.values())

    if is_risky and summary_count >= 2 and career_count == 0:
        return True, "Title/evidence mismatch: honeypot"
    return False, ""


def calculate_stability_advanced(c):
    history = c.get('career_history', [])
    if len(history) < 2:
        return 0.0

    try:
        history_sorted = sorted(history, key=lambda x: x.get('start_date', '9999-12-31'))
    except:
        history_sorted = history

    penalty = 0.0
    short_tenures = 0
    promotions = 0

    for i, h in enumerate(history_sorted):
        months = h.get('duration_months', 0)
        if months == 0:
            return 0.0
        if months < 18:
            short_tenures += 1
        if i > 0 and any(x in h.get('title', '').lower() for x in PROMOTION_TITLES):
            if not any(x in history_sorted[i-1].get('title', '').lower() for x in PROMOTION_TITLES):
                promotions += 1

    if short_tenures >= 2:
        penalty += 0.06
    elif short_tenures == 1:
        penalty += 0.03

    if promotions >= 1:
        penalty -= 0.02

    return max(0, penalty)


def calculate_score_v24(c, concept_embeddings_map, bullet_data, summary_emb, skill_embs, threshold, bm25_index=None, bm25_norm_cap=10.0):
    # `threshold` may be a single float (applied to all categories) or a dict
    # {category: float} for FIX #6 (per-category adaptive thresholds), since
    # different concept categories (e.g. ranking vs. vector DB) tend to have
    # different similarity distributions.
    def cat_threshold(cat):
        return threshold.get(cat, 0.6) if isinstance(threshold, dict) else threshold

    career_evidence = extract_semantic_evidence(bullet_data, concept_embeddings_map, threshold)

    summary_evidence = {cat: [] for cat in JD_CONCEPTS.keys()}
    if summary_emb is not None and len(summary_emb) > 0:
        for category, concept_embs in concept_embeddings_map.items():
            if concept_embs is None or len(concept_embs) == 0:
                continue
            sims = util.cos_sim(summary_emb, concept_embs)[0]
            max_sim = torch.max(sims).item()
            if max_sim > cat_threshold(category):
                summary_evidence[category].append({
                    'quote': c.get('profile', {}).get('summary', '')[:80].strip(),
                    'company': 'Profile',
                    'title': 'Summary',
                    'similarity': max_sim,
                    'company_tier': 1,
                    'duration_months': 0
                })

    is_trap, trap_reason = detect_honeypot(c, career_evidence, summary_evidence)
    has_negative = check_negative_evidence(c, career_evidence)

    # FIX (dilution safeguard for multi-category evidence): since a bullet can now
    # match multiple categories, count how many categories each bullet_idx matched
    # across the whole evidence dict, and divide that bullet's per-category weight
    # by the match count. A bullet matching 4 categories contributes 1/4 weight to
    # each, rather than full weight to all 4 — preventing one generic, broadly
    # worded bullet from dominating the evidence score.
    bullet_match_counts = {}
    for matches in career_evidence.values():
        for m in matches:
            key = m['bullet_idx']
            bullet_match_counts[key] = bullet_match_counts.get(key, 0) + 1

    evidence_score = 0.0
    feature_contribs = {}

    recency_mult = 1.0
    recency_bonus = 0.0
    try:
        end_dates = []
        for h in c.get('career_history', []):
            end_date = h.get('end_date', '')
            if not end_date:
                start_date = h.get('start_date', '')
                months = h.get('duration_months', 0)
                if start_date and months > 0:
                    try:
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_date = (start_dt + timedelta(days=30 * months)).strftime('%Y-%m-%d')
                    except:
                        end_date = ''
            if end_date:
                try:
                    end_dates.append(datetime.strptime(end_date, '%Y-%m-%d'))
                except:
                    try:
                        end_dates.append(datetime.strptime(end_date, '%Y-%m'))
                    except:
                        pass
        if end_dates:
            latest_end_date = max(end_dates)
            age_days = (datetime.today() - latest_end_date).days
            if age_days <= 730:
                recency_mult = 1.10
            elif age_days <= 1825:
                recency_mult = 1.05
            elif age_days <= 2920:
                recency_mult = 1.00
            elif age_days <= 4380:
                recency_mult = 0.95
            else:
                recency_mult = 0.90
            recency_bonus = round(recency_mult - 1.0, 3)
    except:
        recency_mult = 1.0
        recency_bonus = 0.0
    feature_contribs['recency_bonus'] = recency_bonus

    # --- Career evidence scoring ---
    for category, matches in career_evidence.items():
        if matches:
            top_matches = sorted(matches, key=lambda x: x['similarity'] * (1 + x['duration_months']/24), reverse=True)[:2]
            cat_score = 0
            for m in top_matches:
                duration_mult = min(1.5, 1 + m['duration_months']/24)
                tier_mult = 1 + m['company_tier'] / 30  # divisor widened for 5-tier scale (was /20 for 3-tier)
                recency_mult_local = recency_mult
                dilution = bullet_match_counts.get(m['bullet_idx'], 1)
                cat_score += (CATEGORY_WEIGHTS.get(category, 0.03) * m['similarity'] * duration_mult * tier_mult * recency_mult_local) / dilution
            evidence_score += cat_score / len(top_matches)
            feature_contribs[f'evidence_{category}'] = cat_score / len(top_matches)

    # --- Summary evidence scoring (FIX #2: independently capped so a keyword-stuffed
    # summary cannot meaningfully inflate the score; this cap is well below the
    # overall evidence_score ceiling so career evidence remains the dominant signal) ---
    summary_score = 0.0
    for category, matches in summary_evidence.items():
        if matches:
            best_match = max(matches, key=lambda x: x['similarity'])
            summary_score += CATEGORY_WEIGHTS.get(category, 0.03) * best_match['similarity'] * 0.3
    summary_text = c.get('profile', {}).get('summary', '').lower()
    buzzword_terms = ['ai', 'ml', 'llm', 'rag', 'vector', 'embedding', 'gpt', 'genai', 'transformer', 'nlp', 'search', 'ranking', 'retrieval']
    buzzword_count = sum(1 for term in buzzword_terms if term in summary_text)
    career_evidence_count = sum(len(v) for v in career_evidence.values())
    if buzzword_count >= 4 and career_evidence_count <= 2:
        summary_score *= 0.6
    elif buzzword_count >= 3 and career_evidence_count <= 4:
        summary_score *= 0.7
    elif buzzword_count >= 2 and career_evidence_count <= 6:
        summary_score *= 0.8
    elif buzzword_count >= 1 and career_evidence_count <= 1:
        summary_score *= 0.8
    summary_score = min(0.05, summary_score)
    evidence_score += summary_score
    feature_contribs['evidence_summary'] = summary_score
    evidence_score = min(0.35, evidence_score)

    if has_negative:
        evidence_score *= 0.4
    if is_trap:
        evidence_score *= 0.2

    prod_deploy_quotes = set(m['quote'][:30] for m in career_evidence['production_deploy'])
    prod_scale_quotes = set(m['quote'][:30] for m in career_evidence['production_scale'])
    prod_monitor_quotes = set(m['quote'][:30] for m in career_evidence['production_monitor'])

    prod_deploy = min(0.10, len(prod_deploy_quotes) * 0.05)
    prod_scale = min(0.05, len(prod_scale_quotes) * 0.025)
    prod_monitor = min(0.05, len(prod_monitor_quotes) * 0.025)
    feature_contribs['production'] = prod_deploy + prod_scale + prod_monitor

    title = c.get('profile', {}).get('current_title', '').lower()
    role_score = 0.0
    if any(t in title for t in ['engineer', 'developer', 'architect', 'scientist', 'research']):
        role_score = 0.15
    elif 'data' in title and 'analyst' in title:
        role_score = 0.10
    elif 'analyst' in title:
        role_score = 0.05

    if any(t in title for t in RISKY_TITLES) and not any(p in title for p in PRODUCT_MANAGER_TITLES):
        role_score *= 0.3
    feature_contribs['role'] = role_score

    signals = c.get('redrob_signals', {})
    behavior = 0.0
    if signals.get('open_to_work_flag', False):
        behavior += 0.02
    behavior += min(0.03, signals.get('recruiter_response_rate', 0) * 0.03)
    if signals.get('github_activity_score', 0) > 0:
        behavior += min(0.03, signals.get('github_activity_score', 0) / 100 * 0.03)
    if signals.get('interview_completion_rate', 0) > 0.6:
        behavior += 0.02
    behavior = min(0.10, behavior)
    feature_contribs['behavioral'] = behavior

    concept_sims = []
    for cat_embs in concept_embeddings_map.values():
        bullet_sims = []
        for emb, _, _, _, _ in bullet_data:
            bullet_sims.append(util.cos_sim(emb, cat_embs).max().item())
        if bullet_sims:
            concept_sims.append(np.mean(sorted(bullet_sims, reverse=True)[:2]))
    sim_score = np.mean(sorted(concept_sims, reverse=True)[:3]) * 0.10 if concept_sims else 0
    feature_contribs['semantic'] = sim_score

    # --- FIX #4: use the precomputed skill embeddings (semantic match against JD
    # concepts) instead of pure keyword substring matching. Keyword matching is kept
    # as a small secondary signal so exact tool names still count. ---
    skill_score = 0.0
    if skill_embs is not None and len(skill_embs) > 0:
        skill_emb_stack = torch.stack(skill_embs) if isinstance(skill_embs, list) else skill_embs
        best_sims = []
        for cat_embs in concept_embeddings_map.values():
            sims = util.cos_sim(skill_emb_stack, cat_embs)
            best_sims.append(sims.max().item())
        semantic_skill_score = np.mean(sorted(best_sims, reverse=True)[:3]) * 0.04 if best_sims else 0.0
        skill_score += semantic_skill_score

    cand_skills = [s.get('name', '').lower() for s in c.get('skills', [])]
    matched_skills = sum(1 for sk in SKILL_KEYWORDS if any(sk in cs for cs in cand_skills))
    skill_score += min(0.02, matched_skills * 0.002)
    skill_score = min(0.05, skill_score)
    feature_contribs['skills'] = skill_score

    loc_score = 0.0
    feature_contribs['location'] = loc_score

    # --- FIX #1: BM25 normalization replaced. The previous `/10.0` divisor had no
    # theoretical basis. Now we normalize against `bm25_norm_cap`, which is computed
    # upfront from the actual distribution of BM25 max-scores across the candidate
    # pool (95th percentile) — the same pattern already used for the similarity
    # threshold. This makes scores comparable and meaningful relative to the corpus,
    # not an arbitrary fixed constant.
    #
    # FIX #2: tokenization now strips punctuation via regex instead of naive .split(),
    # so tokens like "NDCG," or "BM25:" match cleanly against the JD corpus tokens. ---
    bm25_score = 0.0
    if bm25_index is not None:
        bullet_texts = [b[1] for b in bullet_data]
        summary_text = c.get('profile', {}).get('summary', '')
        skill_names = [s.get('name', '') for s in c.get('skills', [])]
        title_text = c.get('profile', {}).get('current_title', '')
        candidate_text = ' '.join(bullet_texts + [summary_text] + skill_names + [title_text]).lower()
        query_tokens = re.findall(r"\b\w+\b", candidate_text)
        if query_tokens:
            scores = bm25_index.get_scores(query_tokens)  # one score per JD reference sentence/term
            if len(scores) > 0 and scores.max() > 0:
                norm_cap = bm25_norm_cap if bm25_norm_cap > 0 else 10.0
                bm25_score = min(1.0, scores.max() / norm_cap) * 0.05
    feature_contribs['bm25'] = bm25_score

    base_score = evidence_score + prod_deploy + prod_scale + prod_monitor + role_score + behavior + sim_score + skill_score + loc_score + bm25_score

    exp = c.get('profile', {}).get('years_of_experience', 0)
    exp_factor = 0.4 + 0.12 * min(exp, 5)
    exp_factor = min(1.0, exp_factor)
    if exp > 12:
        exp_factor *= 0.9
    feature_contribs['experience_mult'] = exp_factor

    availability_factor = 1.0
    try:
        last_active = datetime.strptime(signals.get('last_active_date', '2000-01-01'), '%Y-%m-%d')
        if last_active < datetime.now() - timedelta(days=180):
            availability_factor *= 0.95
    except:
        pass
    feature_contribs['availability_mult'] = availability_factor

    trust_factor = 1.0
    total_months = 0
    weighted_tier_sum = 0
    for h in c.get('career_history', []):
        months = h.get('duration_months', 0)
        if months > 0:
            tier = get_company_tier(h.get('company', ''))
            weighted_tier_sum += tier * months
            total_months += months

    if total_months > 0:
        avg_tier = weighted_tier_sum / total_months
        trust_factor *= 1 + (avg_tier - 1) * 0.05

    trust_factor *= (1 - calculate_stability_advanced(c))

    if is_trap:
        trust_factor *= 0.2
    feature_contribs['trust_mult'] = trust_factor

    final_score = base_score * exp_factor * availability_factor * trust_factor
    final_score = max(0.001, final_score)

    # --- FIX #3: similarity-weighted confidence instead of a binary matched/unmatched
    # count. A category with a 0.95 similarity match contributes more confidence than
    # one with a 0.63 match, rather than both counting as "1". ---
    matched_sims = []
    for cat in JD_CONCEPTS.keys():
        if career_evidence[cat]:
            if len(career_evidence[cat]) > 1:
                sims = sorted([m['similarity'] for m in career_evidence[cat]], reverse=True)
                matched_sims.append((sims[0] + sims[1]) / 2.0)
            else:
                matched_sims.append(max(m['similarity'] for m in career_evidence[cat]))
        else:
            matched_sims.append(0.0)
    confidence = sum(matched_sims) / len(JD_CONCEPTS)

    return round(final_score, 4), career_evidence, summary_evidence, is_trap, feature_contribs, confidence


def generate_judge_reasoning(c, score, career_evidence, summary_evidence, rank, is_trap, feature_contribs, confidence):
    title = c.get('profile', {}).get('current_title', 'Unknown')
    exp = c.get('profile', {}).get('years_of_experience', 0)

    parts = [f"{title}, {exp}y. Score={score:.3f} Conf={confidence:.0%}"]

    if is_trap:
        parts.append("HONEYPOT: Non-tech title, evidence only in summary.")
        return ' '.join(parts)[:197]

    evidence_cites = []
    strong_categories = []
    for category, matches in career_evidence.items():
        if matches:
            m = max(matches, key=lambda x: x['similarity'] * (1 + x.get('duration_months', 0)/24))
            tier = m.get('company_tier', 1.0)
            tier_str = " [Frontier]" if tier >= 5 else " [Elite]" if tier >= 4 else " [Top]" if tier >= 3 else ""
            quote = m['quote'][:35].strip()
            evidence_cites.append(f'"{quote}"@{m["company"]}{tier_str}')
            strong_categories.append(category)

    if evidence_cites:
        parts.append(f"Evidence: {'; '.join(evidence_cites[:2])}")

    if strong_categories:
        top_categories = strong_categories[:2]
        parts.append(f"Strong: {', '.join(top_categories)}")

    if 'recency_bonus' in feature_contribs:
        recency_bonus = feature_contribs['recency_bonus']
        parts.append(f"Recency:{recency_bonus:+.2f}")
        if recency_bonus > 0:
            parts.append("Recent production experience")
        elif recency_bonus < 0:
            parts.append("Older production experience")

    if 'summary_penalty' in feature_contribs and feature_contribs['summary_penalty'] > 0:
        parts.append("Summary evidence discounted")

    gaps = []
    if not career_evidence["vector_db_prod"]:
        gaps.append('no vector DB')
    if not career_evidence["eval_framework"] and career_evidence["ranking_system"]:
        gaps.append('no NDCG/MRR')
    if c.get('redrob_signals', {}).get('notice_period_days', 0) > 60:
        gaps.append(f"{c['redrob_signals']['notice_period_days']}d notice")

    if gaps:
        parts.append(f"Gaps: {', '.join(gaps[:2])}")

    top_feat = max(feature_contribs.items(), key=lambda x: x[1])
    parts.append(f"Top: {top_feat[0]}={top_feat[1]:.2f}")

    reasoning = ' '.join(parts)[:197]
    return reasoning


# ============================== MAIN EXECUTION ==============================

print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

concept_embeddings_map = {}
for cat, sentences in JD_CONCEPTS.items():
    concept_embeddings_map[cat] = model.encode(sentences, convert_to_tensor=True, device=device)

print("Loading candidates...")
candidates = []
seen_candidate_ids = set()
with open('candidates.jsonl', 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            candidate = json.loads(line)
            cid = candidate.get('candidate_id')
            if cid in seen_candidate_ids:
                print(f"Warning: duplicate candidate_id found: {cid}")
            else:
                seen_candidate_ids.add(cid)
            candidates.append(candidate)
        except:
            continue

print(f"Precomputing BULLET embeddings for {len(candidates)} candidates...")
all_bullet_data = {}
all_summary_texts = []
candidate_ids = []

for c in tqdm(candidates, desc="Extracting bullets"):
    cid = c['candidate_id']
    candidate_ids.append(cid)
    bullets = []
    for h in c.get('career_history', []):
        if len(h.get('description', '')) > 20:
            bullets.append((
                h['description'],
                h.get('company', ''),
                h.get('title', ''),
                h.get('duration_months', 0)
            ))
    all_bullet_data[cid] = bullets
    all_summary_texts.append(c.get('profile', {}).get('summary', ''))

flat_bullets = [(text, comp, title, months, cid) for cid, bullets in all_bullet_data.items()
                for text, comp, title, months in [(b[0], b[1], b[2], b[3]) for b in bullets]]
bullet_texts = [b[0] for b in flat_bullets]
if bullet_texts:
    bullet_embeddings = model.encode(bullet_texts, convert_to_tensor=True, batch_size=256, show_progress_bar=True, device=device)
else:
    bullet_embeddings = None
print(f"Generated embeddings for {len(bullet_texts)} bullets.")

bullet_emb_map = {cid: [] for cid in candidate_ids}
idx = 0
for text, comp, title, months, cid in flat_bullets:
    if bullet_embeddings is None:
        continue
    bullet_emb_map[cid].append((bullet_embeddings[idx], text, comp, title, months))
    idx += 1

summary_embeddings = model.encode(all_summary_texts, convert_to_tensor=True, batch_size=256, show_progress_bar=True, device=device)

flat_skills = [(s.get('name', ''), cid) for cid, c in zip(candidate_ids, candidates) for s in c.get('skills', []) if len(s.get('name', '')) > 1]
skill_texts = [s[0] for s in flat_skills]
if skill_texts:
    skill_embeddings = model.encode(skill_texts, convert_to_tensor=True, batch_size=256, show_progress_bar=True, device=device)
else:
    skill_embeddings = torch.tensor([]).to(device)

skills_emb_map = {cid: [] for cid in candidate_ids}
idx = 0
for skill_text, cid in flat_skills:
    skills_emb_map[cid].append(skill_embeddings[idx])
    idx += 1

summary_emb_map = {cid: emb for cid, emb in zip(candidate_ids, summary_embeddings)}

# --- FIX #1/#5: BM25 corpus is now the JD reference sentences + key technical terms,
# NOT the candidate pool. Each candidate's bullet text becomes the query at scoring
# time (see calculate_score_v24). This replaces the old candidate-indexed corpus
# where get_scores()[0] silently leaked candidate-0's score to everyone. ---
print("Building JD-reference BM25 index...")
jd_reference_sentences = [s.lower() for cat in JD_CONCEPTS.values() for s in cat] + JD_KEY_TERMS
jd_tokenized_corpus = [s.split() for s in jd_reference_sentences]
bm25 = BM25Okapi(jd_tokenized_corpus) if jd_tokenized_corpus else None

print("Computing adaptive similarity thresholds (per category)...")
# FIX #6: compute the adaptive threshold separately per JD category instead of
# pooling all categories together. Ranking-system bullets and vector-DB bullets
# tend to have different similarity score distributions against their respective
# concept sentences, so a single pooled threshold can be too strict for one
# category and too loose for another.
sample_cids = random.sample(candidate_ids, min(1000, len(candidate_ids)))
per_cat_sims = {cat: [] for cat in JD_CONCEPTS.keys()}
for cid in sample_cids:
    for emb, _, _, _, _ in bullet_emb_map[cid]:
        for cat, cat_embs in concept_embeddings_map.items():
            per_cat_sims[cat].append(util.cos_sim(emb, cat_embs).max().item())

threshold = {}
for cat, sims in per_cat_sims.items():
    if sims:
        t = np.percentile(sims, 85)
        threshold[cat] = max(SIM_THRESHOLD_FLOOR, min(0.75, t))
    else:
        threshold[cat] = SIM_THRESHOLD_FLOOR
print("Per-category thresholds:", {k: round(v, 3) for k, v in threshold.items()})

# FIX #1: precompute a BM25 normalization cap from the actual score distribution
# across a sample of candidates (95th percentile of each candidate's max BM25
# score against the JD corpus), instead of an arbitrary fixed divisor.
print("Computing BM25 normalization cap...")
bm25_sample_scores = []
for cid in sample_cids:
    candidate_text = ' '.join([b[1] for b in all_bullet_data[cid]]).lower()
    query_tokens = re.findall(r"\b\w+\b", candidate_text)
    if query_tokens and bm25 is not None:
        s = bm25.get_scores(query_tokens)
        if len(s) > 0:
            bm25_sample_scores.append(s.max())
bm25_norm_cap = np.percentile(bm25_sample_scores, 95) if bm25_sample_scores else 10.0
bm25_norm_cap = max(1e-6, bm25_norm_cap)
print(f"BM25 normalization cap (95th percentile): {bm25_norm_cap:.3f}")

print("Scoring with bullet-level semantic evidence...")
scored = []
for c in tqdm(candidates, desc="Scoring"):
    cid = c['candidate_id']
    try:
        score, career_ev, summary_ev, is_trap, feats, conf = calculate_score_v24(
            c, concept_embeddings_map,
            bullet_emb_map[cid],
            summary_emb_map[cid],
            skills_emb_map[cid],
            threshold,
            bm25,
            bm25_norm_cap
        )
        scored.append({
            'candidate_id': cid,
            'score': score,
            'data': c,
            'career_evidence': career_ev,
            'summary_evidence': summary_ev,
            'is_trap': is_trap,
            'feature_contribs': feats,
            'confidence': conf
        })
    except Exception as e:
        print(f"Error scoring {cid}: {e}")
        scored.append({
            'candidate_id': cid,
            'score': 0.001,
            'data': c,
            'career_evidence': {k: [] for k in JD_CONCEPTS.keys()},
            'summary_evidence': {k: [] for k in JD_CONCEPTS.keys()},
            'is_trap': False,
            'feature_contribs': {},
            'confidence': 0.0
        })

df = pd.DataFrame(scored)
df['evidence_count'] = df['career_evidence'].apply(lambda x: sum(len(v) for v in x.values()))
df['eval_evidence'] = df['career_evidence'].apply(lambda x: len(x.get('eval_framework', [])))
df['prod_evidence'] = df['career_evidence'].apply(lambda x: len(x.get('production_deploy', [])) + len(x.get('production_scale', [])))
df['response'] = df['data'].apply(lambda x: x.get('redrob_signals', {}).get('recruiter_response_rate', 0))
df['interview'] = df['data'].apply(lambda x: x.get('redrob_signals', {}).get('interview_completion_rate', 0))
df['avg_tier'] = df['career_evidence'].apply(lambda x: np.mean([m.get('company_tier', 1.0) for cat in x.values() for m in cat]) if any(x.values()) else 1.0)
df['github'] = df['data'].apply(lambda x: x.get('redrob_signals', {}).get('github_activity_score', 0))

# --- BUG FIX: ascending list now matches the 9 columns in `by` (previously only had
# 4 values, which raises ValueError in pandas). candidate_id is ascending purely for
# deterministic tiebreaking; every ranking signal is descending (higher = better). ---
df = df.sort_values(
    by=['score', 'evidence_count', 'eval_evidence', 'prod_evidence', 'response', 'interview', 'avg_tier', 'github', 'candidate_id'],
    ascending=[False, False, False, False, False, False, False, False, True]
).head(TOP_N).reset_index(drop=True)
df['rank'] = df.index + 1

print("Generating judge-defensible reasoning...")
final = []
for idx, row in df.iterrows():
    reasoning = generate_judge_reasoning(row['data'], row['score'], row['career_evidence'], row['summary_evidence'], row['rank'], row['is_trap'], row['feature_contribs'], row['confidence'])
    final.append({
        'candidate_id': row['candidate_id'],
        'rank': row['rank'],
        'score': row['score'],
        'reasoning': reasoning,
        'confidence': round(row['confidence'], 2)
    })

pd.DataFrame(final).to_csv(f'{TEAM_NAME}.csv', index=False)
print(f"Done. Generated {TEAM_NAME}.csv")
if final:
    print(f"Top 3: {final[0]['candidate_id']}, {final[1]['candidate_id'] if len(final)>1 else 'N/A'}, {final[2]['candidate_id'] if len(final)>2 else 'N/A'}")
