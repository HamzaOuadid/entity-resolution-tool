"""Normalization: standardize name/address fields across sources.

This is deliberately rule-based and inspectable (no ML here) -- the point of
normalization is to make near-duplicate strings *more* alike before we ever
compute a similarity score, so later stages have a cleaner signal to work
with.
"""

from __future__ import annotations

import re
import unicodedata

from .models import NormalizedEntity, RawEntity

# --- legal-suffix normalization -------------------------------------------
# Maps many spellings of a legal form to one canonical short token. Order
# matters: longer/more-specific phrases must be checked before short ones.
_LEGAL_SUFFIXES = [
    (r"\bincorporated\b", "inc"),
    (r"\binc\b\.?", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\b\.?", "corp"),
    (r"\bcompany\b", "co"),
    (r"\bco\b\.?", "co"),
    (r"\blimited liability company\b", "llc"),
    (r"\bl\.?l\.?c\.?\b", "llc"),
    (r"\blimited\b", "ltd"),
    (r"\bltd\b\.?", "ltd"),
    (r"\bp\.?l\.?c\.?\b", "plc"),
    (r"\bb\.?v\.?\b", "bv"),
    (r"\bn\.?v\.?\b", "nv"),
    (r"\bgmbh\b", "gmbh"),
    (r"\bholdings?\b", "holdings"),
    (r"\bgroup\b", "group"),
]

_LEGAL_SUFFIX_TOKENS = {tok for _, tok in _LEGAL_SUFFIXES}

# --- address abbreviation expansion ---------------------------------------
_ADDRESS_ABBREV = {
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "rd": "road",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "mt": "mount",
}

# common company-name word abbreviations that show up across sources
_NAME_WORD_ABBREV = {
    "ind": "industrial",
    "mfg": "manufacturing",
    "tech": "technologies",
    "svcs": "services",
    "intl": "international",
    "natl": "national",
    "assoc": "associates",
    "&": "and",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _basic_clean(text: str) -> str:
    text = _strip_accents(text or "")
    text = text.lower()
    text = text.replace("&", " and ")
    # Drop periods with no replacement (not a space) so dotted abbreviations
    # like "B.V." or "L.L.C." collapse to a single contiguous token ("bv",
    # "llc") instead of being torn into separate one-letter words.
    text = text.replace(".", "")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(raw_name: str) -> tuple[str, str, tuple]:
    """Returns (normalized_name, legal_suffix, sorted_core_tokens).

    normalized_name keeps the legal suffix (as its canonical token) appended
    at the end, so two spellings of the same suffix normalize identically.
    legal_suffix is that canonical token, extracted separately, since
    blocking/scoring often want to compare "the name" without it.
    """
    cleaned = _basic_clean(raw_name)

    found_suffix = ""
    for pattern, canonical in _LEGAL_SUFFIXES:
        if re.search(pattern, cleaned):
            cleaned = re.sub(pattern, " ", cleaned)
            # keep the *first* suffix found in a first pass; but continue
            # stripping others so they don't pollute the core name.
            if not found_suffix:
                found_suffix = canonical
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    words = [
        _NAME_WORD_ABBREV.get(w, w)
        for w in cleaned.split()
        if w not in _LEGAL_SUFFIX_TOKENS
    ]
    core = " ".join(words)
    normalized_name = f"{core} {found_suffix}".strip() if found_suffix else core
    core_tokens = tuple(sorted(set(words)))
    return normalized_name, found_suffix, core_tokens


def normalize_address_text(raw_address: str) -> str:
    cleaned = _basic_clean(raw_address)
    words = [_ADDRESS_ABBREV.get(w, w) for w in cleaned.split()]
    return " ".join(words)


# EU-style ("1017 AB") alternative must be tried before the plain-numeric
# one, otherwise the numeric alternative greedily wins on just the digits
# and the letter suffix is left behind.
_POSTAL_RE = re.compile(r"\b(\d{4}\s?[a-zA-Z]{2}\b|\d{4,5}(?:[- ]?\d{2,4})?\b)")


def extract_postal_code(text: str) -> str:
    """Best-effort postal code extraction from a free-text address blob."""
    if not text:
        return ""
    match = _POSTAL_RE.search(text)
    if not match:
        return ""
    return re.sub(r"[\s-]", "", match.group(1)).upper()


def normalize_postal_code(postal_code: str) -> str:
    if not postal_code:
        return ""
    return re.sub(r"[\s-]", "", postal_code).upper()


def normalize_city(city: str) -> str:
    if not city:
        return ""
    cleaned = _basic_clean(city)
    words = [_ADDRESS_ABBREV.get(w, w) if w != "st" else "saint" for w in cleaned.split()]
    # "st" in a city name (Saint Louis) means "saint", not "street" -- handled
    # by the special case above, distinct from its meaning in a street address.
    return " ".join(words)


def normalize(
    raw_entity: RawEntity,
    city: str = "",
    postal_code: str = "",
) -> NormalizedEntity:
    """normalize(raw_entity) -> NormalizedEntity, per the API contract.

    `city`/`postal_code` are optional pre-parsed fields a source adapter may
    already have (structured sources). If omitted, we try to recover a
    postal code from the free-text address itself.
    """
    normalized_name, legal_suffix, core_tokens = normalize_name(raw_entity.raw_name)
    normalized_address = normalize_address_text(raw_entity.raw_address)
    normalized_postal = normalize_postal_code(postal_code) or extract_postal_code(
        raw_entity.raw_address
    )
    normalized_city = normalize_city(city)

    return NormalizedEntity(
        raw_entity_id=raw_entity.id if raw_entity.id is not None else -1,
        normalized_name=normalized_name,
        normalized_address=normalized_address,
        normalized_city=normalized_city,
        normalized_postal_code=normalized_postal,
        legal_suffix=legal_suffix,
        name_core_tokens=core_tokens,
        source=raw_entity.source,
    )
