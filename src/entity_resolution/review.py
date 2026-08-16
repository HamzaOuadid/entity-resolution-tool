"""Reviewer-facing helpers for the review queue.

The spec's reviewer story is specifically about *surfacing* ambiguity, not
just computing it: "Ambiguous matches are routed to me instead of
auto-decided." That means a reviewer needs enough context to actually make
the call quickly -- not just a bare score. `explain_pair()` breaks the score
down into its component features and flags which ones disagree, since "the
names are identical but the postal codes don't match" is a very different
kind of ambiguity than "the names are a rough match and everything else is
missing."
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .models import CandidatePair, NormalizedEntity
from .scoring import extract_features


@dataclass
class ReviewExplanation:
    pair: CandidatePair
    entity_a: NormalizedEntity
    entity_b: NormalizedEntity
    features: dict
    agreements: list[str]
    disagreements: list[str]
    uncertain: list[str]  # fields missing on one or both sides


_FEATURE_LABELS = {
    "name_token_sort": "name (word-order-independent)",
    "name_partial": "name (substring)",
    "name_jaro_winkler": "name (character-level)",
    "address_token_sort": "street address",
    "postal_match": "postal code",
    "city_similarity": "city",
}


def explain_pair(pair: CandidatePair, entity_a: NormalizedEntity, entity_b: NormalizedEntity) -> ReviewExplanation:
    features = extract_features(entity_a, entity_b)
    agreements, disagreements, uncertain = [], [], []

    for key, value in features.items():
        label = _FEATURE_LABELS[key]
        if key == "address_token_sort" and not (entity_a.normalized_address and entity_b.normalized_address):
            uncertain.append(label)
        elif key == "postal_match" and not (entity_a.normalized_postal_code and entity_b.normalized_postal_code):
            uncertain.append(label)
        elif key == "city_similarity" and not (entity_a.normalized_city and entity_b.normalized_city):
            uncertain.append(label)
        elif value >= 0.85:
            agreements.append(label)
        elif value <= 0.5:
            disagreements.append(label)

    return ReviewExplanation(
        pair=pair, entity_a=entity_a, entity_b=entity_b, features=features,
        agreements=agreements, disagreements=disagreements, uncertain=uncertain,
    )


def format_explanation(exp: ReviewExplanation) -> str:
    a, b, pair = exp.entity_a, exp.entity_b, exp.pair
    lines = [
        f"--- pair_id={pair.id}  score={pair.similarity_score:.3f} (review band) ---",
        f"  A [{a.source}]: {a.normalized_name} | {a.normalized_address} | {a.normalized_city} {a.normalized_postal_code}",
        f"  B [{b.source}]: {b.normalized_name} | {b.normalized_address} | {b.normalized_city} {b.normalized_postal_code}",
    ]
    if exp.agreements:
        lines.append(f"  agrees on:     {', '.join(exp.agreements)}")
    if exp.disagreements:
        lines.append(f"  disagrees on:  {', '.join(exp.disagreements)}")
    if exp.uncertain:
        lines.append(f"  missing data:  {', '.join(exp.uncertain)}")
    return "\n".join(lines)


def export_review_queue_csv(explanations: list[ReviewExplanation], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "score", "source_a", "name_a", "address_a", "source_b", "name_b", "address_b",
            "agrees_on", "disagrees_on", "missing_data", "reviewer_decision",
        ])
        for exp in explanations:
            a, b, pair = exp.entity_a, exp.entity_b, exp.pair
            writer.writerow([
                pair.id, f"{pair.similarity_score:.3f}",
                a.source, a.normalized_name, f"{a.normalized_address}, {a.normalized_city} {a.normalized_postal_code}",
                b.source, b.normalized_name, f"{b.normalized_address}, {b.normalized_city} {b.normalized_postal_code}",
                "; ".join(exp.agreements), "; ".join(exp.disagreements), "; ".join(exp.uncertain),
                "",  # left blank for the reviewer to fill in
            ])
    return out_path
