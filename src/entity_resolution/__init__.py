"""entity_resolution: record-linkage pipeline for messy multi-source entity data.

Pipeline stages, matching the spec's API contract:
    normalize(raw_entity) -> NormalizedEntity
    block(entities)       -> list[CandidatePair]
    score(pair)           -> float
    resolve(pairs)        -> {matches, no_matches, review_queue}
"""

__version__ = "0.1.0"
