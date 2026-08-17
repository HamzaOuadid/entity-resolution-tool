"""End-to-end pipeline tests against the synthetic messy dataset -- this is
the core proof-of-value: real precision/recall/F1 against known ground
truth, blocking recall measured separately, and explicit demonstration of
the review-queue band and the franchise-sibling false-positive check.
"""

from entity_resolution.datagen import generate_dataset
from entity_resolution.pipeline import run_pipeline
from entity_resolution.resolve import ResolveThresholds


def test_pipeline_achieves_reasonable_precision_and_recall():
    dataset = generate_dataset(n_entities=300, seed=42)
    result = run_pipeline(dataset)
    m = result.metrics
    assert m["full_dataset_precision"] >= 0.85, m
    assert m["full_dataset_recall"] >= 0.60, m
    assert m["blocking_recall"] >= 0.85, m


def test_review_queue_is_non_empty_and_contains_real_ambiguous_pairs():
    """Every review-queue pair got there for one of two legitimate reasons:
    (a) its own score fell in the mid band, or (b) it's part of a cluster
    flagged for conflicting evidence (see resolve.py's transitive-chain
    safeguard) -- which can pull in an individually high- or low-scoring
    pair. Either way, it must not have been silently auto-decided."""
    dataset = generate_dataset(n_entities=300, seed=42)
    result = run_pipeline(dataset)
    assert len(result.resolve_result.review_queue) > 0

    t = ResolveThresholds()
    flagged_ids = {
        eid for c in result.resolve_result.clusters if c.status == "flagged_conflict" for eid in c.entity_ids
    }
    for p in result.resolve_result.review_queue:
        assert p.similarity_score is not None
        in_mid_band = t.no_match_threshold < p.similarity_score < t.match_threshold
        in_flagged_cluster = p.entity_a_id in flagged_ids or p.entity_b_id in flagged_ids
        assert in_mid_band or in_flagged_cluster, (
            f"pair {p.id} (score={p.similarity_score}) is in review_queue but is neither "
            "mid-band nor part of a flagged-conflict cluster"
        )


def test_review_queue_pairs_are_not_double_counted_as_matches_or_no_matches():
    dataset = generate_dataset(n_entities=300, seed=42)
    result = run_pipeline(dataset)
    match_ids = {p.id for p in result.resolve_result.matches}
    no_match_ids = {p.id for p in result.resolve_result.no_matches}
    review_ids = {p.id for p in result.resolve_result.review_queue}
    assert not (match_ids & review_ids)
    assert not (no_match_ids & review_ids)
    assert not (match_ids & no_match_ids)


def test_confusable_franchise_entities_are_not_auto_merged():
    """Edge case from the spec: legitimate distinct entities with very
    similar names (franchise locations) must not be over-merged."""
    dataset = generate_dataset(n_entities=200, seed=42, confusable_fraction=0.25)
    result = run_pipeline(dataset)

    confusable_true_ids = {g.true_entity_id for g in dataset.ground_truth if g.is_confusable_distinct}
    # map true_entity_id -> set of normalized-entity ids
    true_to_ids: dict = {}
    for eid, true_id in result.id_to_true.items():
        if true_id in confusable_true_ids:
            true_to_ids.setdefault(true_id, set()).add(eid)

    # build cluster membership: entity id -> cluster index
    entity_to_cluster = {}
    for i, cluster in enumerate(result.resolve_result.clusters):
        for eid in cluster.entity_ids:
            entity_to_cluster[eid] = i

    sibling_groups: dict = {}
    for g in dataset.ground_truth:
        if g.is_confusable_distinct:
            sibling_groups.setdefault(g.true_entity_id, True)

    # For each pair of DIFFERENT true entities that are confusable siblings,
    # their members must not end up in the same resolved (non-conflicted) cluster.
    false_merges = 0
    true_ids = list(true_to_ids.keys())
    for i in range(len(true_ids)):
        for j in range(i + 1, len(true_ids)):
            ids_i = true_to_ids[true_ids[i]]
            ids_j = true_to_ids[true_ids[j]]
            clusters_i = {entity_to_cluster[e] for e in ids_i if e in entity_to_cluster}
            clusters_j = {entity_to_cluster[e] for e in ids_j if e in entity_to_cluster}
            shared = clusters_i & clusters_j
            for cidx in shared:
                if result.resolve_result.clusters[cidx].status == "resolved":
                    false_merges += 1

    assert false_merges == 0, f"{false_merges} confusable franchise siblings were auto-merged"


def test_blocking_recall_measured_separately_from_match_recall():
    dataset = generate_dataset(n_entities=250, seed=7)
    result = run_pipeline(dataset)
    assert "blocking_recall" in result.metrics
    assert "full_dataset_recall" in result.metrics
    # they are genuinely different numbers computed by different code paths
    assert result.metrics["blocking_recall"] >= result.metrics["full_dataset_recall"] - 0.15


def test_hand_label_sample_precision_recall_are_reported():
    dataset = generate_dataset(n_entities=300, seed=42)
    result = run_pipeline(dataset, sample_size=200)
    m = result.metrics
    assert m["hand_label_sample_size"] > 0
    assert 0.0 <= m["hand_label_precision"] <= 1.0
    assert 0.0 <= m["hand_label_recall"] <= 1.0
    assert result.hand_label_sample is not None


def test_missing_fields_across_dataset_do_not_crash_pipeline():
    dataset = generate_dataset(n_entities=150, seed=42, missing_field_rate=0.4)
    result = run_pipeline(dataset)
    assert result.metrics["n_entities"] == len(dataset.raw_entities)


def test_number_of_duplicate_entities_correctly_merged_is_reportable():
    """Definition-of-done metric: report the number of duplicate entities
    correctly merged."""
    dataset = generate_dataset(n_entities=300, seed=42)
    result = run_pipeline(dataset)
    correctly_merged = sum(
        1 for c in result.resolve_result.clusters
        if c.status == "resolved" and len(c.entity_ids) >= 2
    )
    assert correctly_merged > 0
