"""Blocking: cheap keys that avoid full pairwise (O(n^2)) comparison.

A single blocking strategy is fragile -- e.g. "first 3 letters of the name"
misses a match when the very first letters are typo'd. So we combine several
cheap, independent strategies and take the *union* of the pairs each one
proposes. A true match only needs to survive in *one* strategy to become a
candidate pair; that's what gives blocking good recall while still cutting
the comparison volume by orders of magnitude versus all-pairs.

Strategies:
  1. name-prefix + postal-prefix : first 3 chars of the name's core tokens,
     joined, + first 3 chars of the postal code (falls back to city).
  2. sorted-initials + city      : sorted first letters of each significant
     word (order-independent) + normalized city -- survives word reordering.
  3. phonetic (Soundex) + city   : survives common spelling/typo variation
     that changes the first letters (e.g. "Xavier" vs "Zavier").

Each strategy groups normalized entities by key, then emits all pairs within
a group. Groups larger than `max_block_size` are skipped for that strategy
(a huge block is a sign of a low-information key, e.g. "" or a very generic
key) -- this bounds worst-case cost.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import jellyfish

from .models import CandidatePair, NormalizedEntity


def _name_prefix_key(entity: NormalizedEntity) -> str:
    tokens = entity.name_core_tokens or tuple(entity.normalized_name.split())
    joined = "".join(tokens)[:3]
    postal = (entity.normalized_postal_code or "")[:3]
    geo = postal or entity.normalized_city[:3]
    if not joined or not geo:
        return ""
    return f"np:{joined}|{geo}"


def _sorted_initials_key(entity: NormalizedEntity) -> str:
    tokens = entity.name_core_tokens or tuple(entity.normalized_name.split())
    initials = "".join(sorted(t[0] for t in tokens if t))
    if not initials or not entity.normalized_city:
        return ""
    return f"si:{initials}|{entity.normalized_city}"


def _phonetic_key(entity: NormalizedEntity) -> str:
    tokens = entity.name_core_tokens or tuple(entity.normalized_name.split())
    if not tokens or not entity.normalized_city:
        return ""
    codes = sorted(jellyfish.soundex(t) for t in tokens if t)
    return f"ph:{''.join(codes)}|{entity.normalized_city}"


STRATEGIES = {
    "name_prefix_postal": _name_prefix_key,
    "sorted_initials_city": _sorted_initials_key,
    "phonetic_city": _phonetic_key,
}


def block(
    entities: list[NormalizedEntity],
    max_block_size: int = 60,
    strategies: dict | None = None,
) -> list[CandidatePair]:
    """block(entities) -> list[CandidatePair], per the API contract.

    Cross-source pairs are what we actually care about (within-source
    duplicates shouldn't normally occur since sources dedupe internally),
    but we don't hard-exclude same-source pairs -- a source can still have
    accidental duplicates, and excluding them would hide that edge case.
    """
    strategies = strategies or STRATEGIES
    seen_pairs: dict[tuple[int, int], str] = {}

    for strategy_name, key_fn in strategies.items():
        groups: dict[str, list[NormalizedEntity]] = defaultdict(list)
        for entity in entities:
            key = key_fn(entity)
            if key:
                groups[key].append(entity)

        for key, members in groups.items():
            if len(members) < 2 or len(members) > max_block_size:
                continue
            for a, b in combinations(members, 2):
                if a.id == b.id:
                    continue
                pair_key = tuple(sorted((a.id, b.id)))
                if pair_key not in seen_pairs:
                    seen_pairs[pair_key] = key

    return [
        CandidatePair(entity_a_id=a, entity_b_id=b, blocking_key=key)
        for (a, b), key in seen_pairs.items()
    ]


def blocking_stats(entities: list[NormalizedEntity], max_block_size: int = 60) -> dict:
    """Diagnostic: how many comparisons blocking produces vs. full pairwise."""
    n = len(entities)
    full_pairwise = n * (n - 1) // 2
    pairs = block(entities, max_block_size=max_block_size)
    reduction = 1 - (len(pairs) / full_pairwise) if full_pairwise else 0.0
    return {
        "n_entities": n,
        "full_pairwise_comparisons": full_pairwise,
        "candidate_pairs": len(pairs),
        "reduction_ratio": reduction,
    }
