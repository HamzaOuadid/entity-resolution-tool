"""End-to-end orchestration: normalize -> block -> score -> resolve -> evaluate.

This is the glue used by both the CLI and the test suite so the "real
demo run" the README quotes numbers from is the exact same code path the
tests exercise -- no separate, unverified demo script.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .blocking import block as block_fn
from .datagen import GeneratedDataset
from .evaluation import (
    HandLabelSample,
    PRF,
    blocking_recall,
    evaluate_against_hand_labels,
    hand_label_sample,
    pairwise_prf,
    true_entity_map,
    true_match_pairs,
)
from .models import NormalizedEntity
from .normalize import normalize as normalize_fn
from .resolve import ResolveResult, ResolveThresholds, resolve as resolve_fn
from .scoring import HeuristicScorer, score_all


@dataclass
class PipelineResult:
    normalized_entities: list[NormalizedEntity]
    entities_by_id: dict
    candidate_pairs: list
    resolve_result: ResolveResult
    id_to_true: dict
    metrics: dict = field(default_factory=dict)
    hand_label_sample: HandLabelSample | None = None


def normalize_dataset(dataset: GeneratedDataset) -> tuple[list[NormalizedEntity], dict]:
    """Assigns sequential ids and normalizes every raw entity.

    Returns (normalized_entities, id_to_true) where id_to_true maps
    normalized-entity id -> ground-truth true_entity_id.
    """
    true_lookup = {(g.source, g.external_id): g.true_entity_id for g in dataset.ground_truth}

    normalized = []
    id_to_true = {}
    for i, raw in enumerate(dataset.raw_entities):
        entity_id = i + 1
        raw_with_id = dataclasses.replace(raw, id=entity_id)
        norm = normalize_fn(raw_with_id)
        norm = dataclasses.replace(norm, id=entity_id)
        normalized.append(norm)
        id_to_true[entity_id] = true_lookup[(raw.source, raw.external_id)]

    return normalized, id_to_true


def run_pipeline(
    dataset: GeneratedDataset,
    thresholds: ResolveThresholds | None = None,
    scorer=None,
    max_block_size: int = 60,
    sample_size: int = 300,
    compute_hand_label_metrics: bool = True,
) -> PipelineResult:
    thresholds = thresholds or ResolveThresholds()
    scorer = scorer or HeuristicScorer()

    normalized, id_to_true = normalize_dataset(dataset)
    entities_by_id = {e.id: e for e in normalized}

    candidate_pairs = block_fn(normalized, max_block_size=max_block_size)
    candidate_pairs = score_all(candidate_pairs, entities_by_id, scorer=scorer)
    # assign synthetic pair ids (in-memory pipeline doesn't require a DB)
    candidate_pairs = [dataclasses.replace(p, id=i + 1) for i, p in enumerate(candidate_pairs)]

    resolve_result = resolve_fn(candidate_pairs, thresholds=thresholds)

    true_pairs = true_match_pairs(id_to_true)
    b_recall = blocking_recall(candidate_pairs, true_pairs)

    predicted_match_pairs = {
        (min(p.entity_a_id, p.entity_b_id), max(p.entity_a_id, p.entity_b_id)) for p in resolve_result.matches
    }
    full_prf = pairwise_prf(predicted_match_pairs, true_pairs)

    metrics = {
        "n_entities": len(normalized),
        "n_candidate_pairs": len(candidate_pairs),
        "n_true_match_pairs": len(true_pairs),
        "blocking_recall": b_recall.recall,
        "blocking_missed": b_recall.fn,
        "full_dataset_precision": full_prf.precision,
        "full_dataset_recall": full_prf.recall,
        "full_dataset_f1": full_prf.f1,
        "n_matches": len(resolve_result.matches),
        "n_no_matches": len(resolve_result.no_matches),
        "n_review_queue": len(resolve_result.review_queue),
        "n_clusters": len(resolve_result.clusters),
        "n_flagged_conflict_clusters": sum(1 for c in resolve_result.clusters if c.status == "flagged_conflict"),
    }

    sample = None
    if compute_hand_label_metrics:
        sample = hand_label_sample(candidate_pairs, id_to_true, sample_size=sample_size)
        sample_prf = evaluate_against_hand_labels(resolve_result, sample)
        metrics.update({
            "hand_label_sample_size": len(sample.pairs),
            "hand_label_precision": sample_prf.precision,
            "hand_label_recall": sample_prf.recall,
            "hand_label_f1": sample_prf.f1,
            "hand_label_tp": sample_prf.tp,
            "hand_label_fp": sample_prf.fp,
            "hand_label_fn": sample_prf.fn,
            "hand_label_tn": sample_prf.tn,
        })

    return PipelineResult(
        normalized_entities=normalized,
        entities_by_id=entities_by_id,
        candidate_pairs=candidate_pairs,
        resolve_result=resolve_result,
        id_to_true=id_to_true,
        metrics=metrics,
        hand_label_sample=sample,
    )
