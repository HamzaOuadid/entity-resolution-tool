"""resolve(pairs) -> {matches, no_matches, review_queue}, per the API contract.

Two thresholds carve the score range into three bands:

    score >= match_threshold        -> "match"
    score <= no_match_threshold     -> "no_match"
    otherwise                       -> "review"   (the ambiguous middle band)

The review band is a first-class output, not a fallback: it is exactly what
the spec's reviewer story asks for -- ambiguous pairs are *surfaced*, not
silently auto-merged and not silently dropped.

Clustering & the transitive-chain edge case
--------------------------------------------
A "match" decision on pairs (A,B) and (B,C) implies A, B, C are the same
entity (transitivity) -- so matched pairs are merged into clusters with
union-find. But naive single-link transitive closure has a known failure
mode: if A~B and B~C both score above threshold, but A and C individually
look nothing alike (or were never even compared, or *were* compared and
scored as a firm no_match), chaining still merges all three. That's a real
false-merge risk, not a hypothetical one.

Mitigation implemented here: after forming connected components from
"match" edges, each cluster is checked for *internal conflicts* against
every scored pair we actually have for member pairs within it. If any two
members of a would-be cluster were themselves scored as "no_match", the
cluster has contradictory evidence and is downgraded: it is broken out of
the auto-merge path and every pair touching the conflicting members is
routed to the review queue instead. This does not guarantee no bad merges
(two members might simply never have been compared -- blocking may not have
paired them directly), but it catches the case where we have *explicit*
contradicting evidence, which is the case the spec's edge case is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import CandidatePair, Decision


@dataclass
class ResolveThresholds:
    # Tuned by sweeping both thresholds against the synthetic dataset's
    # hand-labelled sample (see README for the sweep table). 0.75/0.60 sits
    # near the full-dataset F1 peak (0.889) while keeping the review band a
    # manageable size relative to a looser no_match_threshold of 0.55 (same
    # F1, ~25% more pairs pushed to review for no precision/recall gain).
    match_threshold: float = 0.75
    no_match_threshold: float = 0.60


@dataclass
class ClusterResult:
    entity_ids: frozenset
    status: str  # "resolved" | "flagged_conflict"


@dataclass
class ResolveResult:
    matches: list[CandidatePair] = field(default_factory=list)
    no_matches: list[CandidatePair] = field(default_factory=list)
    review_queue: list[CandidatePair] = field(default_factory=list)
    clusters: list[ClusterResult] = field(default_factory=list)


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def classify(pair: CandidatePair, thresholds: ResolveThresholds) -> Decision:
    if pair.similarity_score is None:
        raise ValueError(f"pair {pair.id} has no similarity_score; run score_all() first")
    if pair.similarity_score >= thresholds.match_threshold:
        return "match"
    if pair.similarity_score <= thresholds.no_match_threshold:
        return "no_match"
    return "review"


def cluster_matches(matches: list[CandidatePair]) -> list[frozenset]:
    """Transitive closure over 'match' edges via union-find."""
    ids = set()
    for p in matches:
        ids.add(p.entity_a_id)
        ids.add(p.entity_b_id)
    uf = _UnionFind(ids)
    for p in matches:
        uf.union(p.entity_a_id, p.entity_b_id)

    groups: dict = {}
    for i in ids:
        groups.setdefault(uf.find(i), set()).add(i)
    return [frozenset(g) for g in groups.values()]


def resolve(
    pairs: list[CandidatePair],
    thresholds: ResolveThresholds | None = None,
) -> ResolveResult:
    """resolve(pairs) -> ResolveResult(matches, no_matches, review_queue, clusters).

    Every pair in `pairs` must already have similarity_score set (run
    scoring.score_all() first).
    """
    thresholds = thresholds or ResolveThresholds()

    matches, no_matches, review = [], [], []
    for pair in pairs:
        decision = classify(pair, thresholds)
        if decision == "match":
            matches.append(pair)
        elif decision == "no_match":
            no_matches.append(pair)
        else:
            review.append(pair)

    raw_clusters = cluster_matches(matches)

    # conflict check: does any no_match pair contradict a proposed cluster?
    no_match_pairs = {(min(p.entity_a_id, p.entity_b_id), max(p.entity_a_id, p.entity_b_id)) for p in no_matches}

    clusters: list[ClusterResult] = []
    conflicted_members: set = set()
    for members in raw_clusters:
        if len(members) < 2:
            continue
        has_conflict = False
        member_list = sorted(members)
        for i in range(len(member_list)):
            for j in range(i + 1, len(member_list)):
                if (member_list[i], member_list[j]) in no_match_pairs:
                    has_conflict = True
                    break
            if has_conflict:
                break
        status = "flagged_conflict" if has_conflict else "resolved"
        clusters.append(ClusterResult(entity_ids=members, status=status))
        if has_conflict:
            conflicted_members |= members

    if conflicted_members:
        # demote match/no_match decisions touching a conflicted cluster to
        # review, and pull them out of matches/no_matches accordingly.
        kept_matches, kept_no_matches = [], []
        for p in matches:
            if p.entity_a_id in conflicted_members or p.entity_b_id in conflicted_members:
                review.append(p)
            else:
                kept_matches.append(p)
        for p in no_matches:
            if p.entity_a_id in conflicted_members or p.entity_b_id in conflicted_members:
                review.append(p)
            else:
                kept_no_matches.append(p)
        matches, no_matches = kept_matches, kept_no_matches

    return ResolveResult(matches=matches, no_matches=no_matches, review_queue=review, clusters=clusters)
