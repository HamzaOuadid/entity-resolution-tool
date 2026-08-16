"""Synthetic messy multi-source entity data generator.

Why synthetic instead of a downloaded public dataset: two public datasets
with genuine, *known* entity overlap and a redistribution-clean license are
surprisingly hard to find and pin (see README "Risks / Open Questions"). The
task explicitly directs generating realistic messy synthetic data instead,
since we then know ground truth exactly and can publish *real* precision/
recall numbers instead of estimates.

We generate three source exports with three different schemas (mirroring
real multi-source messiness: a government-style business registry, a
commercial directory with a single free-text address field, and a CRM
export), covering one base set of "true" entities:

  * Most entities appear in 2-3 of the sources, each appearance independently
    mangled (abbreviations, typos, dropped legal suffix, reordered words,
    missing fields).
  * A subset of entities are "confusable siblings": distinct real entities
    (think franchise locations) that share a near-identical name but differ
    in address/city -- these must NOT be merged, and exist specifically to
    stress-test false positives.

Ground truth (source, external_id) -> true_entity_id is written separately
and is used only for evaluation, never fed into normalize/block/score.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from .models import RawEntity

ADJECTIVES = [
    "Acme", "Northgate", "Bluewater", "Summit", "Ironwood", "Silverline",
    "Meridian", "Falcon", "Harborview", "Union", "Vantage", "Redstone",
    "Crestline", "Whitfield", "Eastbrook", "Pinecrest", "Anchor", "Highland",
    "Coastal", "Sterling", "Ridgeline", "Cobalt", "Lakeside", "Fairmont",
    "Granite", "Brookhaven", "Westgate", "Amberfield", "Cedarbrook", "Novara",
    "Thornbury", "Kingsley", "Wexford", "Hollowell", "Marlow", "Delacroix",
    "Ashworth", "Brambleton", "Foxhollow", "Greystone",
]

INDUSTRY_WORDS = [
    "Industrial", "Logistics", "Manufacturing", "Technologies", "Consulting",
    "Foods", "Energy", "Materials", "Freight", "Analytics", "Systems",
    "Textiles", "Chemicals", "Robotics", "Agriculture", "Media", "Pharma",
    "Construction", "Automotive", "Software", "Auto Repair", "Coffee",
    "Pizza", "Hardware", "Bakery", "Electric", "Plumbing", "Landscaping",
]

US_SUFFIXES = ["Inc.", "Inc", "Corporation", "Corp.", "LLC", "Co.", "Company"]
EU_SUFFIXES = ["B.V.", "BV", "GmbH", "N.V.", "Ltd."]
GROUP_QUALIFIERS = ["Holdings", "Group", ""]

STREET_NAMES = [
    "Maple", "Oak", "Cedar", "Elm", "Washington", "Lincoln", "Church",
    "Main", "Park", "Union", "Franklin", "Sunset", "Highland", "River",
    "Industrial Parkway", "Harbor", "Market", "Mill", "Prinsengracht",
    "Kerkstraat", "Bahnhofstrasse",
]
STREET_TYPES_US = ["St", "Ave", "Rd", "Blvd", "Dr", "Ln", "Ct"]

US_CITIES = [
    ("Denver", "CO", "80202"), ("Austin", "TX", "73301"), ("Portland", "OR", "97201"),
    ("Columbus", "OH", "43215"), ("Raleigh", "NC", "27601"), ("Tampa", "FL", "33602"),
    ("Boise", "ID", "83702"), ("Madison", "WI", "53703"), ("Reno", "NV", "89501"),
    ("Tulsa", "OK", "74103"), ("Spokane", "WA", "99201"), ("Wichita", "KS", "67202"),
]
EU_CITIES = [
    ("Amsterdam", "1017 AB"), ("Rotterdam", "3011 CE"), ("Utrecht", "3511 LN"),
    ("Munich", "80331"), ("Hamburg", "20095"), ("Antwerp", "2000"),
    ("Brussels", "1000"), ("Eindhoven", "5611 EM"), ("Leiden", "2311 GJ"),
]

_KEYBOARD_NEIGHBORS = {
    "a": "qs", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


@dataclass
class GroundTruthRow:
    source: str
    external_id: str
    true_entity_id: str
    is_confusable_distinct: bool


@dataclass
class GeneratedDataset:
    raw_entities: list[RawEntity]
    ground_truth: list[GroundTruthRow]
    # per-source rows in the raw (pre-RawEntity) shape, so the CLI can dump
    # them into the three differently-shaped CSV schemas.
    source_rows: dict = field(default_factory=dict)


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    op = rng.choice(["swap", "delete", "insert", "substitute"])
    i = rng.randrange(1, len(word) - 1)
    if op == "swap" and i < len(word) - 1:
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rng.choice(string.ascii_lowercase) + word[i:]
    if op == "substitute":
        neighbors = _KEYBOARD_NEIGHBORS.get(word[i].lower(), word[i])
        return word[:i] + rng.choice(neighbors) + word[i + 1:]
    return word


def _mutate_name(name: str, suffix: str, rng: random.Random) -> str:
    """Apply a random, source-realistic mutation to a canonical company name."""
    choice = rng.random()
    words = name.split()
    if choice < 0.20:
        # drop the legal suffix entirely
        return name[: -(len(suffix) + 1)] if suffix and name.endswith(suffix) else name
    if choice < 0.35 and suffix:
        # alternate spelling of the suffix
        alt = {
            "Inc.": "Inc", "Inc": "Incorporated", "Corp.": "Corporation",
            "LLC": "L.L.C.", "Co.": "Company", "B.V.": "BV", "BV": "B.V.",
            "Ltd.": "Limited", "GmbH": "GmbH", "N.V.": "NV",
        }.get(suffix, suffix)
        return " ".join(words[:-1] + [alt])
    if choice < 0.50 and len(words) > 2:
        # word-order shuffle of the non-suffix words (tests token-sort robustness)
        core, tail = words[:-1], words[-1:]
        rng.shuffle(core)
        return " ".join(core + tail)
    if choice < 0.65:
        # abbreviate an industry word
        abbrev = {
            "Industrial": "Ind.", "Manufacturing": "Mfg.", "Technologies": "Tech.",
            "International": "Intl.", "Associates": "Assoc.", "Services": "Svcs.",
        }
        return " ".join(abbrev.get(w, w) for w in words)
    if choice < 0.80:
        # character-level typo in one word
        idx = rng.randrange(len(words))
        words[idx] = _typo(words[idx], rng)
        return " ".join(words)
    if choice < 0.90:
        return name.upper()
    return name  # exact


def _mutate_street(street: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        street = _typo(street, rng)
    return street


def _region_for(rng: random.Random) -> str:
    return "us" if rng.random() < 0.6 else "eu"


def _make_base_entity(rng: random.Random, idx: int, name_override: str | None = None) -> dict:
    region = _region_for(rng)
    adj = rng.choice(ADJECTIVES)
    noun = rng.choice(INDUSTRY_WORDS)
    qualifier = rng.choice(GROUP_QUALIFIERS)
    suffix = rng.choice(US_SUFFIXES if region == "us" else EU_SUFFIXES)
    parts = [name_override or adj, noun]
    if qualifier:
        parts.append(qualifier)
    parts.append(suffix)
    canonical_name = " ".join(parts)

    street_num = rng.randint(10, 9999)
    street = rng.choice(STREET_NAMES)
    if region == "us":
        street_type = rng.choice(STREET_TYPES_US)
        city, state, postal = rng.choice(US_CITIES)
        address = f"{street_num} {street} {street_type}, {city}, {state} {postal}"
    else:
        city, postal = rng.choice(EU_CITIES)
        state = ""
        address = f"{street} {street_num}, {postal} {city}"

    return {
        "entity_id": f"E{idx:04d}",
        "canonical_name": canonical_name,
        "suffix": suffix,
        "street_num": street_num,
        "street": street,
        "street_type": locals().get("street_type", ""),
        "city": city,
        "state": state,
        "postal": postal,
        "region": region,
    }


SOURCES = ["registry", "directory", "crm"]


def generate_dataset(
    n_entities: int = 350,
    confusable_fraction: float = 0.12,
    seed: int = 42,
    missing_field_rate: float = 0.12,
) -> GeneratedDataset:
    """Generate a messy multi-source synthetic dataset with known ground truth.

    Returns raw entities already collapsed into the common RawEntity shape
    (source, external_id, raw_name, raw_address), plus per-source structured
    rows (source_rows) suitable for writing out as three differently-shaped
    CSVs, plus the ground-truth entity_id each row truly belongs to.
    """
    rng = random.Random(seed)
    base_entities: list[dict] = []
    idx = 0

    n_confusable_groups = int(n_entities * confusable_fraction / 2)
    n_plain = n_entities - n_confusable_groups * 2

    for _ in range(n_plain):
        base_entities.append(_make_base_entity(rng, idx))
        idx += 1

    # confusable sibling groups: same "brand" adjective+noun, different
    # cities -- legitimately distinct entities with very similar names.
    for _ in range(n_confusable_groups):
        adj = rng.choice(ADJECTIVES)
        noun = rng.choice(["Auto Repair", "Coffee", "Pizza", "Hardware", "Bakery"])
        siblings = rng.sample(US_CITIES, 2)
        for city, state, postal in siblings:
            e = _make_base_entity(rng, idx, name_override=adj)
            # force the noun and city so the two siblings share a brand name
            # but sit in different, unambiguous locations.
            suffix = rng.choice(["", "LLC", "Inc."])
            name = f"{adj} {noun}" + (f" {suffix}" if suffix else "")
            e["canonical_name"] = name
            e["suffix"] = suffix
            e["city"], e["state"], e["postal"] = city, state, postal
            e["region"] = "us"
            street_num = rng.randint(10, 9999)
            street = rng.choice(STREET_NAMES)
            street_type = rng.choice(STREET_TYPES_US)
            e["street_num"], e["street"], e["street_type"] = street_num, street, street_type
            e["is_confusable"] = True
            e["sibling_group"] = f"{adj}-{noun}"
            base_entities.append(e)
            idx += 1

    for e in base_entities:
        e.setdefault("is_confusable", False)

    raw_entities: list[RawEntity] = []
    ground_truth: list[GroundTruthRow] = []
    source_rows: dict[str, list[dict]] = {s: [] for s in SOURCES}
    ext_counters = {s: 0 for s in SOURCES}

    for e in base_entities:
        n_appearances = rng.choice([2, 2, 3, 3, 4]) if not e["is_confusable"] else rng.choice([2, 3])
        sources_for_entity = rng.sample(SOURCES, k=min(n_appearances, len(SOURCES)))
        if n_appearances > len(SOURCES):
            sources_for_entity += rng.choices(SOURCES, k=n_appearances - len(SOURCES))

        for source in sources_for_entity:
            ext_counters[source] += 1
            external_id = f"{source[:3].upper()}-{ext_counters[source]:05d}"

            mutated_name = _mutate_name(e["canonical_name"], e["suffix"], rng)
            street = _mutate_street(e["street"], rng)
            drop_postal = rng.random() < missing_field_rate
            drop_city = rng.random() < (missing_field_rate * 0.4)

            if e["region"] == "us":
                street_type = e.get("street_type") or rng.choice(STREET_TYPES_US)
                full_street = f"{e['street_num']} {street} {street_type}"
                city = "" if drop_city else e["city"]
                postal = "" if drop_postal else e["postal"]
            else:
                full_street = f"{street} {e['street_num']}"
                city = "" if drop_city else e["city"]
                postal = "" if drop_postal else e["postal"]

            row = {
                "external_id": external_id,
                "name": mutated_name,
                "street": full_street,
                "city": city,
                "postal": postal,
                "state": e.get("state", ""),
                "country": "US" if e["region"] == "us" else "EU",
            }
            source_rows[source].append(row)

            if source == "registry":
                raw_address = ", ".join(p for p in [full_street, city, postal] if p)
            elif source == "directory":
                raw_address = ", ".join(p for p in [full_street, city, postal] if p)
            else:
                raw_address = ", ".join(p for p in [full_street, city] if p)

            raw_entities.append(
                RawEntity(
                    source=source,
                    external_id=external_id,
                    raw_name=mutated_name,
                    raw_address=raw_address,
                )
            )
            ground_truth.append(
                GroundTruthRow(
                    source=source,
                    external_id=external_id,
                    true_entity_id=e["entity_id"],
                    is_confusable_distinct=e["is_confusable"],
                )
            )

    return GeneratedDataset(
        raw_entities=raw_entities, ground_truth=ground_truth, source_rows=source_rows
    )
