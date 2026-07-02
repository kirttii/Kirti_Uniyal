"""
Validation script for candidate_scoring_v24.py output.

Run this AFTER generating Kirti_Uniyal.csv (the v24 output) to inspect:
  - score distribution
  - confidence distribution
  - sanity flags (zero-confidence top-N entries, honeypots that slipped through)
  - (optional) a diff against a previous version's output CSV, to see how
    many candidates moved in/out of the top 100 and by how much

This is meant to be run before adding a reranker stage, per the recommended
validation sequence: validate the embedding-layer fixes in isolation first,
so any quality change after adding a reranker can be attributed correctly.

Usage:
    python validate_v24.py Kirti_Uniyal.csv [previous_version.csv]
"""
import sys
import pandas as pd
import numpy as np


def describe_distribution(series, name):
    print(f"\n--- {name} distribution ---")
    print(f"  count:  {len(series)}")
    print(f"  mean:   {series.mean():.4f}")
    print(f"  median: {series.median():.4f}")
    print(f"  std:    {series.std():.4f}")
    print(f"  min:    {series.min():.4f}")
    print(f"  max:    {series.max():.4f}")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  p{p}: {np.percentile(series, p):.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_v24.py <output_csv> [previous_version_csv]")
        sys.exit(1)

    current_path = sys.argv[1]
    prev_path = sys.argv[2] if len(sys.argv) > 2 else None

    df = pd.read_csv(current_path)
    print(f"Loaded {len(df)} ranked candidates from {current_path}")

    # --- Score and confidence distributions ---
    describe_distribution(df['score'], 'Score')
    describe_distribution(df['confidence'], 'Confidence')

    # --- Sanity flags ---
    print("\n--- Sanity checks ---")
    zero_conf = (df['confidence'] == 0).sum()
    print(f"  Candidates in top-{len(df)} with confidence == 0: {zero_conf}")
    if zero_conf > 0:
        print("  -> WARNING: these candidates ranked into the top N with no matched")
        print("     evidence categories at all. Worth spot-checking a few rows; this")
        print("     can happen if behavioral/role/skill signals alone pushed them up.")

    honeypot_flags = df['reasoning'].str.contains('HONEYPOT', na=False).sum()
    print(f"  Honeypot-flagged candidates still in top-{len(df)}: {honeypot_flags}")
    if honeypot_flags > 0:
        print("  -> WARNING: honeypot detection should usually suppress these out of")
        print("     the top N; if they're still appearing, other signals are too strong")
        print("     relative to the honeypot penalty multiplier.")

    low_score_high_rank = df[(df['rank'] <= 20) & (df['score'] < df['score'].median())]
    print(f"  Top-20 candidates scoring below the overall median: {len(low_score_high_rank)}")

    # --- Top 10 preview ---
    print("\n--- Top 10 candidates ---")
    preview_cols = [c for c in ['rank', 'candidate_id', 'score', 'confidence', 'reasoning'] if c in df.columns]
    with pd.option_context('display.max_colwidth', 80):
        print(df[preview_cols].head(10).to_string(index=False))

    # --- Comparison against a previous version, if provided ---
    if prev_path:
        prev_df = pd.read_csv(prev_path)
        print(f"\nLoaded {len(prev_df)} ranked candidates from {prev_path} for comparison")

        cur_ids = set(df['candidate_id'])
        prev_ids = set(prev_df['candidate_id'])

        new_entrants = cur_ids - prev_ids
        dropped = prev_ids - cur_ids
        stable = cur_ids & prev_ids

        print("\n--- Top-N overlap ---")
        print(f"  Stable (in both top-N lists): {len(stable)}")
        print(f"  New entrants (in current, not previous): {len(new_entrants)}")
        print(f"  Dropped (in previous, not current): {len(dropped)}")
        print(f"  Overlap ratio: {len(stable) / max(len(cur_ids), 1):.1%}")

        if stable:
            cur_ranks = df.set_index('candidate_id')['rank']
            prev_ranks = prev_df.set_index('candidate_id')['rank']
            rank_deltas = pd.Series([prev_ranks[cid] - cur_ranks[cid] for cid in stable])
            print("\n--- Rank movement for stable candidates (positive = moved up) ---")
            print(f"  mean delta:   {rank_deltas.mean():.2f}")
            print(f"  median delta: {rank_deltas.median():.2f}")
            print(f"  max improvement: {rank_deltas.max()}")
            print(f"  max drop: {rank_deltas.min()}")

        # Score comparison for stable candidates
        if stable and 'score' in prev_df.columns:
            cur_scores = df.set_index('candidate_id')['score']
            prev_scores = prev_df.set_index('candidate_id')['score']
            score_deltas = pd.Series([cur_scores[cid] - prev_scores[cid] for cid in stable])
            print("\n--- Score delta for stable candidates (current - previous) ---")
            print(f"  mean delta:   {score_deltas.mean():.4f}")
            print(f"  median delta: {score_deltas.median():.4f}")
            pct_increased = (score_deltas > 0).mean()
            print(f"  % with increased score: {pct_increased:.1%}")
            if pct_increased > 0.85:
                print("  -> WARNING: if almost everyone's score went up, the change may be")
                print("     inflating scores broadly (e.g. multi-category bullet credit)")
                print("     rather than improving relative ranking quality. Check whether")
                print("     RELATIVE ORDER changed meaningfully, not just absolute scores.")

    print("\nValidation complete.")


if __name__ == '__main__':
    main()
