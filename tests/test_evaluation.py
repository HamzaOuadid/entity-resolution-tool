from entity_resolution.evaluation import (
    hand_label_sample,
    pairwise_prf,
    true_match_pairs,
)
from entity_resolution.models import CandidatePair


def test_true_match_pairs_groups_by_true_entity_id():
    id_to_true = {1: "E1", 2: "E1", 3: "E1", 4: "E2", 5: "E3"}
    pairs = true_match_pairs(id_to_true)
    assert pairs == {(1, 2), (1, 3), (2, 3)}


def test_pairwise_prf_perfect_prediction():
    true_pairs = {(1, 2), (3, 4)}
    predicted = {(1, 2), (3, 4)}
    result = pairwise_prf(predicted, true_pairs)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_pairwise_prf_penalizes_false_positive():
    true_pairs = {(1, 2)}
    predicted = {(1, 2), (5, 6)}  # (5,6) is a false merge
    result = pairwise_prf(predicted, true_pairs)
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.tp == 1
    assert result.fp == 1


def test_pairwise_prf_penalizes_false_negative():
    true_pairs = {(1, 2), (3, 4)}
    predicted = {(1, 2)}  # missed (3,4)
    result = pairwise_prf(predicted, true_pairs)
    assert result.recall == 0.5
    assert result.fn == 1


def test_hand_label_sample_stratifies_across_bands():
    pairs = (
        [CandidatePair(id=i, entity_a_id=i, entity_b_id=i + 1000, blocking_key="k", similarity_score=0.95) for i in range(50)]
        + [CandidatePair(id=100 + i, entity_a_id=100 + i, entity_b_id=1100 + i, blocking_key="k", similarity_score=0.65) for i in range(20)]
        + [CandidatePair(id=200 + i, entity_a_id=200 + i, entity_b_id=1200 + i, blocking_key="k", similarity_score=0.1) for i in range(50)]
    )
    id_to_true = {}
    sample = hand_label_sample(pairs, id_to_true, sample_size=30)
    assert len(sample.pairs) <= 30
    assert len(sample.pairs) > 0
    # mid-band pairs (the ambiguous ones) must be represented
    mid_ids = {100 + i for i in range(20)}
    sampled_ids = {p.id for p in sample.pairs}
    assert sampled_ids & mid_ids
