import dataclasses

from entity_resolution.blocking import block, blocking_stats
from entity_resolution.datagen import generate_dataset
from entity_resolution.normalize import normalize


def _normalized_from_generated(n_entities=120, seed=13, **kwargs):
    dataset = generate_dataset(n_entities=n_entities, seed=seed, **kwargs)
    entities = []
    for i, raw in enumerate(dataset.raw_entities):
        raw_with_id = dataclasses.replace(raw, id=i + 1)
        norm = normalize(raw_with_id)
        norm = dataclasses.replace(norm, id=i + 1)
        entities.append(norm)
    return entities, dataset


def test_block_reduces_comparisons_versus_full_pairwise():
    entities, _ = _normalized_from_generated(n_entities=200, seed=21)
    stats = blocking_stats(entities)
    assert stats["candidate_pairs"] < stats["full_pairwise_comparisons"]
    assert stats["reduction_ratio"] > 0.5


def test_block_never_pairs_an_entity_with_itself():
    entities, _ = _normalized_from_generated(n_entities=80, seed=5)
    pairs = block(entities)
    for p in pairs:
        assert p.entity_a_id != p.entity_b_id


def test_block_pairs_are_deduplicated_and_ordered():
    entities, _ = _normalized_from_generated(n_entities=80, seed=5)
    pairs = block(entities)
    seen = set()
    for p in pairs:
        assert p.entity_a_id < p.entity_b_id
        key = (p.entity_a_id, p.entity_b_id)
        assert key not in seen
        seen.add(key)


def test_blocking_recall_on_known_true_matches():
    """Core edge case from the spec: measure blocking recall separately --
    do known true-duplicate pairs survive blocking?"""
    entities, dataset = _normalized_from_generated(n_entities=250, seed=33)
    id_to_raw_idx = {e.id: e.id - 1 for e in entities}  # raw index == id-1 by construction

    true_entity_of = {}
    for g in dataset.ground_truth:
        true_entity_of[(g.source, g.external_id)] = g.true_entity_id

    # map entity.id -> true_entity_id via the raw entity it was built from
    id_to_true = {}
    for idx, raw in enumerate(dataset.raw_entities):
        id_to_true[idx + 1] = true_entity_of[(raw.source, raw.external_id)]

    from collections import defaultdict
    groups = defaultdict(list)
    for eid, true_id in id_to_true.items():
        groups[true_id].append(eid)

    true_pairs = set()
    for true_id, members in groups.items():
        if len(members) >= 2:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    true_pairs.add(tuple(sorted((members[i], members[j]))))

    pairs = block(entities)
    candidate_pairs_set = {(p.entity_a_id, p.entity_b_id) for p in pairs}
    survived = sum(1 for tp in true_pairs if tp in candidate_pairs_set)
    blocking_recall = survived / len(true_pairs) if true_pairs else 0.0

    assert len(true_pairs) > 10  # sanity: dataset actually has duplicates
    assert blocking_recall >= 0.85, f"blocking recall too low: {blocking_recall:.3f}"


def test_a_typo_riddled_true_match_can_still_be_missed_by_a_single_strategy():
    """Edge case: a single blocking strategy (name-prefix) can legitimately
    miss a true match when the first letters are typo'd -- that's exactly
    why we combine multiple independent strategies."""
    from entity_resolution.blocking import _name_prefix_key
    from entity_resolution.models import NormalizedEntity

    a = NormalizedEntity(
        id=1, raw_entity_id=1, normalized_name="acme industrial",
        normalized_address="", normalized_city="denver",
        normalized_postal_code="80202", name_core_tokens=("acme", "industrial"),
    )
    b = NormalizedEntity(
        id=2, raw_entity_id=2, normalized_name="zcme industrial",  # typo'd first letter
        normalized_address="", normalized_city="denver",
        normalized_postal_code="90000",  # also a different postal code
        name_core_tokens=("zcme", "industrial"),
    )
    assert _name_prefix_key(a) != _name_prefix_key(b)
