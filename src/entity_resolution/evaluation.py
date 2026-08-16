"""Evaluation against known ground truth: precision/recall/F1, blocking
recall, and a stratified "hand-labelled sample" workflow.

Honesty note (see README): because the dataset is synthetic and *we*
generated it, we know true duplicate clusters exactly. There is no human in
the loop labeling pairs. `hand_label_sample()` simulates the spec's
"hand-label a sample of candidate pairs" step by drawing a stratified sample
across score bands and assigning each pair's label from ground truth --
i.e. what a perfectly accurate human reviewer would have said. This gives
real, reproducible precision/recall numbers rather than fabricated ones, at
the cost of not exercising actual human-labeling noise/disagreement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .datagen import GroundTruthRow
from .models import CandidatePair


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int = 0


def true_entity_map(ground_truth: list[GroundTruthRow], raw_id_lookup: dict) -> dict[int, str]:
    """Maps normalized_entity id -> true_entity_id, given a lookup from
    (source, external_id) -> entity id."""
    result = {}
    for g in ground_truth:
        eid = raw_id_lookup.get((g.source, g.external_id))
        if eid is not None:
            result[eid] = g.true_entity_id
    return result


def true_match_pairs(id_to_true: dict[int, str]) -> set[tuple[int, int]]:
    from collections import defaultdict

    groups = defaultdict(list)
    for entity_id, true_id in id_to_true.items():
        groups[true_id].append(entity_id)
    pairs = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def blocking_recall(candidate_pairs: list[CandidatePair], true_pairs: set[tuple[int, int]]) -> PRF:
    candidate_set = {(min(p.entity_a_id, p.entity_b_id), max(p.entity_a_id, p.entity_b_id)) for p in candidate_pairs}
    tp = len(true_pairs & candidate_set)
    fn = len(true_pairs - candidate_set)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return PRF(precision=float("nan"), recall=recall, f1=float("nan"), tp=tp, fp=0, fn=fn)


def pairwise_prf(
    predicted_match_pairs: set[tuple[int, int]],
    true_pairs: set[tuple[int, int]],
    universe: set[tuple[int, int]] | None = None,
) -> PRF:
    """Precision/recall/F1 of predicted-match pairs against true-match pairs.

    If `universe` (the set of pairs actually under consideration, e.g. a
    hand-labelled sample or all candidate pairs) is given, negatives are
    counted only within it -- otherwise recall/precision are computed from
    the pair sets directly (fp/fn well-defined; tn is not, since the
    universe of all non-pairs is enormous and not meaningful here).
    """
    tp = len(predicted_match_pairs & true_pairs)
    fp = len(predicted_match_pairs - true_pairs)
    fn = len(true_pairs - predicted_match_pairs)
    tn = 0
    if universe is not None:
        tn = len(universe - predicted_match_pairs - true_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


@dataclass
class HandLabelSample:
    pairs: list[CandidatePair]
    labels: dict[int, str]  # pair.id -> 'match' | 'no_match'


def hand_label_sample(
    candidate_pairs: list[CandidatePair],
    id_to_true: dict[int, str],
    sample_size: int = 300,
    seed: int = 99,
) -> HandLabelSample:
    """Stratified sample across score bands, labeled from ground truth.

    Stratifying by score band (rather than pure random sampling) ensures the
    sample actually contains ambiguous review-band pairs and clear matches/
    non-matches in reasonable proportion, instead of being dominated by the
    (usually huge) obvious-no-match tail.
    """
    rng = random.Random(seed)

    def band(p: CandidatePair) -> str:
        s = p.similarity_score or 0.0
        if s >= 0.83:
            return "high"
        if s <= 0.55:
            return "low"
        return "mid"

    by_band: dict[str, list[CandidatePair]] = {"high": [], "mid": [], "low": []}
    for p in candidate_pairs:
        by_band[band(p)].append(p)

    # take everything from "mid" (usually smallest and most interesting),
    # then fill the rest proportionally from high/low.
    target_mid = min(len(by_band["mid"]), sample_size // 3)
    sampled = rng.sample(by_band["mid"], target_mid) if by_band["mid"] else []
    remaining = sample_size - len(sampled)
    pool = by_band["high"] + by_band["low"]
    rng.shuffle(pool)
    sampled += pool[:remaining]

    labels = {}
    for p in sampled:
        true_a = id_to_true.get(p.entity_a_id)
        true_b = id_to_true.get(p.entity_b_id)
        labels[p.id] = "match" if (true_a is not None and true_a == true_b) else "no_match"

    return HandLabelSample(pairs=sampled, labels=labels)


def evaluate_against_hand_labels(
    resolve_result,
    sample: HandLabelSample,
) -> PRF:
    """Precision/recall of the pipeline's match decisions, restricted to the
    hand-labelled sample (what the spec calls for: publish P/R on the
    hand-labelled sample specifically)."""
    sample_pair_ids = {p.id for p in sample.pairs}
    predicted_match_ids = {p.id for p in resolve_result.matches} & sample_pair_ids
    true_match_ids = {pid for pid, label in sample.labels.items() if label == "match"}

    tp = len(predicted_match_ids & true_match_ids)
    fp = len(predicted_match_ids - true_match_ids)
    fn = len(true_match_ids - predicted_match_ids)
    tn = len(sample_pair_ids - predicted_match_ids - true_match_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)
