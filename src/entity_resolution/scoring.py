"""Pairwise scoring: turn a CandidatePair into a similarity score in [0, 1].

Two scorers are provided behind the same `score(pair)` interface:

  * HeuristicScorer -- a transparent, hand-weighted combination of string-
    similarity features (rapidfuzz token-sort / partial ratio / Jaro-Winkler,
    postal-code and city agreement). No training data required, fully
    inspectable. This is the default/production scorer.
  * LogisticScorer  -- the same features, but the weights are learned from a
    *training* split of labeled pairs via scikit-learn's LogisticRegression,
    the same idea `dedupe` uses internally. Evaluated only on a held-out
    *test* split so numbers aren't inflated by fitting on what we test on.

We didn't reach for Splink or `dedupe` directly: Splink pulls in DuckDB/Spark
backends and `dedupe` needs a C-extension build toolchain, both heavy for a
demo of this size. The feature-engineering + logistic-regression pattern
underneath `dedupe` is straightforward to reproduce with rapidfuzz +
scikit-learn (already available), so that's what LogisticScorer does.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .models import CandidatePair, NormalizedEntity

FEATURE_NAMES = [
    "name_token_sort",
    "name_partial",
    "name_jaro_winkler",
    "address_token_sort",
    "postal_match",
    "city_similarity",
]


def extract_features(a: NormalizedEntity, b: NormalizedEntity) -> dict[str, float]:
    name_token_sort = fuzz.token_sort_ratio(a.normalized_name, b.normalized_name) / 100.0
    name_partial = fuzz.partial_ratio(a.normalized_name, b.normalized_name) / 100.0
    name_jw = JaroWinkler.similarity(a.normalized_name, b.normalized_name)

    if a.normalized_address and b.normalized_address:
        address_token_sort = fuzz.token_sort_ratio(a.normalized_address, b.normalized_address) / 100.0
    else:
        address_token_sort = 0.0  # missing field: no evidence either way

    if a.normalized_postal_code and b.normalized_postal_code:
        postal_match = 1.0 if a.normalized_postal_code == b.normalized_postal_code else 0.0
    else:
        postal_match = 0.5  # unknown -- neither confirms nor denies

    if a.normalized_city and b.normalized_city:
        city_similarity = fuzz.ratio(a.normalized_city, b.normalized_city) / 100.0
    else:
        city_similarity = 0.5

    return {
        "name_token_sort": name_token_sort,
        "name_partial": name_partial,
        "name_jaro_winkler": name_jw,
        "address_token_sort": address_token_sort,
        "postal_match": postal_match,
        "city_similarity": city_similarity,
    }


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]


# Hand-tuned weights: name similarity dominates (it's the strongest signal
# and the only one guaranteed to be present), postal/city corroborate but
# are down-weighted because they're frequently missing (see normalize.py --
# missing fields score a neutral 0.5, so their weighted contribution shrinks
# toward the midpoint rather than dragging the score down outright).
_HEURISTIC_WEIGHTS = {
    "name_token_sort": 0.34,
    "name_partial": 0.10,
    "name_jaro_winkler": 0.26,
    "address_token_sort": 0.12,
    "postal_match": 0.12,
    "city_similarity": 0.06,
}


class HeuristicScorer:
    """Deterministic weighted-feature scorer. No fitting, no leakage risk."""

    name = "heuristic"

    def score_entities(self, a: NormalizedEntity, b: NormalizedEntity) -> float:
        features = extract_features(a, b)
        return sum(_HEURISTIC_WEIGHTS[k] * v for k, v in features.items())

    def score_pair(self, pair: CandidatePair, entities_by_id: dict[int, NormalizedEntity]) -> float:
        a = entities_by_id[pair.entity_a_id]
        b = entities_by_id[pair.entity_b_id]
        return self.score_entities(a, b)


@dataclass
class LabeledExample:
    entity_a_id: int
    entity_b_id: int
    label: int  # 1 = match, 0 = no_match


class LogisticScorer:
    """Learned scorer: LogisticRegression over the same feature set.

    Must be `fit()` on a training split before use. `score_entities` returns
    the model's predicted probability of a match.
    """

    name = "logistic"

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(
        self,
        examples: list[LabeledExample],
        entities_by_id: dict[int, NormalizedEntity],
    ) -> None:
        X, y = [], []
        for ex in examples:
            a = entities_by_id[ex.entity_a_id]
            b = entities_by_id[ex.entity_b_id]
            X.append(feature_vector(extract_features(a, b)))
            y.append(ex.label)
        self._model.fit(X, y)
        self._fitted = True

    def score_entities(self, a: NormalizedEntity, b: NormalizedEntity) -> float:
        if not self._fitted:
            raise RuntimeError("LogisticScorer.fit() must be called before scoring")
        x = [feature_vector(extract_features(a, b))]
        return float(self._model.predict_proba(x)[0][1])

    def score_pair(self, pair: CandidatePair, entities_by_id: dict[int, NormalizedEntity]) -> float:
        a = entities_by_id[pair.entity_a_id]
        b = entities_by_id[pair.entity_b_id]
        return self.score_entities(a, b)


def score(pair: CandidatePair, entities_by_id: dict[int, NormalizedEntity], scorer=None) -> float:
    """score(pair) -> float, per the API contract. Defaults to HeuristicScorer."""
    scorer = scorer or HeuristicScorer()
    return scorer.score_pair(pair, entities_by_id)


def score_all(
    pairs: list[CandidatePair],
    entities_by_id: dict[int, NormalizedEntity],
    scorer=None,
) -> list[CandidatePair]:
    """Returns new CandidatePair objects with similarity_score populated."""
    import dataclasses

    scorer = scorer or HeuristicScorer()
    scored = []
    for pair in pairs:
        s = scorer.score_pair(pair, entities_by_id)
        scored.append(dataclasses.replace(pair, similarity_score=s))
    return scored
