from entity_resolution.models import NormalizedEntity
from entity_resolution.scoring import HeuristicScorer, LogisticScorer, LabeledExample, extract_features


def _entity(id_, name, city="denver", postal="80202", address="123 main street"):
    return NormalizedEntity(
        id=id_, raw_entity_id=id_, normalized_name=name, normalized_address=address,
        normalized_city=city, normalized_postal_code=postal,
        name_core_tokens=tuple(sorted(name.split())),
    )


def test_identical_entities_score_near_one():
    a = _entity(1, "acme industrial holdings bv")
    b = _entity(2, "acme industrial holdings bv")
    score = HeuristicScorer().score_entities(a, b)
    assert score > 0.95


def test_unrelated_entities_score_low():
    a = _entity(1, "acme industrial holdings bv", city="denver", postal="80202", address="500 oak street")
    b = _entity(2, "brookhaven pizza llc", city="austin", postal="73301", address="17 elm road")
    score = HeuristicScorer().score_entities(a, b)
    assert score < 0.45


def test_near_duplicate_with_typo_scores_high():
    a = _entity(1, "acme industrial holdings bv")
    b = _entity(2, "acme industrial holdngs bv")  # dropped a letter
    score = HeuristicScorer().score_entities(a, b)
    assert score > 0.83


def test_confusable_franchise_siblings_score_low_despite_similar_name():
    """Edge case: same brand name, different city/postal -- must not score
    as a confident match on name similarity alone."""
    a = _entity(1, "northgate pizza", city="denver", postal="80202", address="500 oak street")
    b = _entity(2, "northgate pizza", city="austin", postal="73301", address="17 elm road")
    score = HeuristicScorer().score_entities(a, b)
    # identical name, but address disagreement should pull the score down
    # out of the confident-match range
    assert score < 0.83


def test_missing_fields_do_not_crash_scoring():
    a = _entity(1, "acme industrial holdings bv", city="", postal="", address="")
    b = _entity(2, "acme industrial holdings bv", city="denver", postal="80202")
    score = HeuristicScorer().score_entities(a, b)
    assert 0.0 <= score <= 1.0


def test_extract_features_returns_all_expected_keys():
    a = _entity(1, "acme industrial")
    b = _entity(2, "acme industrial")
    features = extract_features(a, b)
    assert set(features) == {
        "name_token_sort", "name_partial", "name_jaro_winkler",
        "address_token_sort", "postal_match", "city_similarity",
    }


def test_logistic_scorer_learns_to_separate_matches_from_non_matches():
    entities = {
        1: _entity(1, "acme industrial holdings bv", city="amsterdam", postal="1017ab"),
        2: _entity(2, "acme industrial holdngs bv", city="amsterdam", postal="1017ab"),  # true match
        3: _entity(3, "brookhaven pizza llc", city="austin", postal="73301"),  # unrelated
    }
    examples = [
        LabeledExample(1, 2, label=1),
        LabeledExample(1, 3, label=0),
        LabeledExample(2, 3, label=0),
    ]
    scorer = LogisticScorer()
    scorer.fit(examples, entities)
    match_score = scorer.score_entities(entities[1], entities[2])
    nonmatch_score = scorer.score_entities(entities[1], entities[3])
    assert match_score > nonmatch_score
