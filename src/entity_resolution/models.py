"""Data model, matching the spec's data-model section.

Naming note vs. the spec's literal column names: `raw_entities` has a
`source_id` column meaning "the ID assigned by the source system" (e.g. the
CRM's own record id). `normalized_entities` also has a `source_id` column,
but the spec means "the id of the row it was normalized from" -- i.e. a
foreign key into `raw_entities`. Those are two different things with the same
name, which is confusing to implement literally. We disambiguate:

  * raw_entities.external_id   == spec's raw_entities.source_id
  * normalized_entities.raw_entity_id == spec's normalized_entities.source_id

Everything else follows the spec's column names and types exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Decision = Literal["match", "no_match", "review"]


@dataclass(frozen=True)
class RawEntity:
    """One row as it appeared in a source system, before normalization."""

    source: str  # which source system this came from, e.g. "registry"
    external_id: str  # the source system's own record id
    raw_name: str
    raw_address: str
    id: Optional[int] = None  # raw_entities.id once persisted


@dataclass(frozen=True)
class NormalizedEntity:
    """A cleaned, comparable view of a RawEntity."""

    raw_entity_id: int  # FK -> raw_entities.id (spec: normalized_entities.source_id)
    normalized_name: str
    normalized_address: str
    normalized_city: str = ""
    normalized_postal_code: str = ""
    legal_suffix: str = ""  # e.g. "bv", "inc", "llc" (stripped from the name)
    name_core_tokens: tuple = field(default_factory=tuple)  # sorted significant tokens
    source: str = ""
    id: Optional[int] = None


@dataclass(frozen=True)
class CandidatePair:
    """A pair of entities worth comparing, produced by blocking."""

    entity_a_id: int  # normalized_entities.id
    entity_b_id: int
    blocking_key: str
    similarity_score: Optional[float] = None
    id: Optional[int] = None


@dataclass
class ResolutionDecision:
    """The outcome for one candidate pair."""

    pair_id: int
    decision: Decision
    reviewer: Optional[str] = None
    hand_label: Optional[str] = None  # 'match' | 'no_match', from ground truth / a human
    id: Optional[int] = None
