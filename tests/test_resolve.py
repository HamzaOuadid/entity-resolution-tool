from entity_resolution.models import CandidatePair
from entity_resolution.resolve import ResolveThresholds, classify, cluster_matches, resolve


def _pair(id_, a, b, score):
    return CandidatePair(id=id_, entity_a_id=a, entity_b_id=b, blocking_key="k", similarity_score=score)


def test_classify_bands():
    t = ResolveThresholds(match_threshold=0.8, no_match_threshold=0.5)
    assert classify(_pair(1, 1, 2, 0.95), t) == "match"
    assert classify(_pair(2, 1, 2, 0.2), t) == "no_match"
    assert classify(_pair(3, 1, 2, 0.65), t) == "review"
    # boundary values
    assert classify(_pair(4, 1, 2, 0.8), t) == "match"
    assert classify(_pair(5, 1, 2, 0.5), t) == "no_match"


def test_resolve_routes_pairs_into_three_bands():
    t = ResolveThresholds(match_threshold=0.8, no_match_threshold=0.5)
    pairs = [_pair(1, 1, 2, 0.95), _pair(2, 3, 4, 0.1), _pair(3, 5, 6, 0.65)]
    result = resolve(pairs, thresholds=t)
    assert len(result.matches) == 1
    assert len(result.no_matches) == 1
    assert len(result.review_queue) == 1


def test_review_queue_pairs_are_neither_merged_nor_dropped():
    t = ResolveThresholds(match_threshold=0.8, no_match_threshold=0.5)
    pairs = [_pair(1, 1, 2, 0.65)]
    result = resolve(pairs, thresholds=t)
    assert result.matches == []
    assert result.no_matches == []
    assert len(result.review_queue) == 1
    assert result.review_queue[0].id == 1


def test_transitive_closure_merges_a_match_chain():
    t = ResolveThresholds(match_threshold=0.8, no_match_threshold=0.5)
    pairs = [_pair(1, 1, 2, 0.9), _pair(2, 2, 3, 0.9)]
    result = resolve(pairs, thresholds=t)
    assert len(result.clusters) == 1
    assert result.clusters[0].entity_ids == frozenset({1, 2, 3})
    assert result.clusters[0].status == "resolved"


def test_transitive_chain_with_explicit_conflict_is_flagged_not_silently_merged():
    """Edge case from the spec: transitive-match chaining risk. A~B and B~C
    both score as matches, but A vs C was itself scored as a firm no_match --
    that's contradictory evidence, and the cluster must be flagged for
    review rather than silently merged."""
    t = ResolveThresholds(match_threshold=0.8, no_match_threshold=0.5)
    pairs = [
        _pair(1, 1, 2, 0.9),   # A~B: match
        _pair(2, 2, 3, 0.9),   # B~C: match
        _pair(3, 1, 3, 0.1),   # A~C: firm no_match -- contradicts the chain
    ]
    result = resolve(pairs, thresholds=t)
    assert len(result.clusters) == 1
    assert result.clusters[0].status == "flagged_conflict"
    # none of the three touched pairs should have ended up silently
    # decided as match or no_match
    assert result.matches == []
    assert result.no_matches == []
    assert len(result.review_queue) == 3


def test_cluster_matches_is_pure_union_find():
    pairs = [_pair(1, 1, 2, 0.9), _pair(2, 3, 4, 0.9)]
    clusters = cluster_matches(pairs)
    assert frozenset({1, 2}) in clusters
    assert frozenset({3, 4}) in clusters
    assert len(clusters) == 2


def test_resolve_raises_if_pair_not_scored():
    unscored = CandidatePair(id=1, entity_a_id=1, entity_b_id=2, blocking_key="k")
    try:
        resolve([unscored])
        assert False, "expected ValueError"
    except ValueError:
        pass
