"""Per-source CSV I/O.

Each of the three synthetic sources uses a genuinely different schema, the
way real multi-source data does:

  * registry.csv  -- government-style export, structured fields, incl. a
                      free-standing incorporation_year column we ignore.
  * directory.csv -- commercial directory export, a single free-text
                      full_address field (no separate city/postal columns).
  * crm.csv       -- CRM export, Salesforce-style "billing_*" field names.

Loading a source means adapting its schema into the common RawEntity shape
(source, external_id, raw_name, raw_address), while keeping any already-
structured city/postal-code fields around so normalize() doesn't have to
re-derive them from free text when it doesn't need to.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import NamedTuple

from .datagen import GeneratedDataset, GroundTruthRow
from .models import RawEntity

REGISTRY_FIELDS = [
    "reg_id", "legal_name", "street_address", "city", "postal_code",
    "country", "incorporation_year",
]
DIRECTORY_FIELDS = ["dir_id", "company_name", "full_address", "phone", "category"]
CRM_FIELDS = ["crm_id", "account_name", "billing_street", "billing_city", "billing_zip", "industry"]


class LoadedEntity(NamedTuple):
    raw_entity: RawEntity
    city: str
    postal_code: str


def write_dataset(dataset: GeneratedDataset, out_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: dict[str, Path] = {}

    registry_path = out_dir / "source_registry.csv"
    with registry_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in dataset.source_rows.get("registry", []):
            street = ", ".join(p for p in [row["street"]] if p)
            writer.writerow({
                "reg_id": row["external_id"],
                "legal_name": row["name"],
                "street_address": street,
                "city": row["city"],
                "postal_code": row["postal"],
                "country": row["country"],
                "incorporation_year": rng.randint(1985, 2023),
            })
    paths["registry"] = registry_path

    directory_path = out_dir / "source_directory.csv"
    with directory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIRECTORY_FIELDS)
        writer.writeheader()
        for row in dataset.source_rows.get("directory", []):
            full_address = ", ".join(p for p in [row["street"], row["city"], row["postal"]] if p)
            writer.writerow({
                "dir_id": row["external_id"],
                "company_name": row["name"],
                "full_address": full_address,
                "phone": f"+1-555-{rng.randint(1000,9999)}",
                "category": "business",
            })
    paths["directory"] = directory_path

    crm_path = out_dir / "source_crm.csv"
    with crm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CRM_FIELDS)
        writer.writeheader()
        for row in dataset.source_rows.get("crm", []):
            writer.writerow({
                "crm_id": row["external_id"],
                "account_name": row["name"],
                "billing_street": row["street"],
                "billing_city": row["city"],
                "billing_zip": row["postal"],
                "industry": "unknown",
            })
    paths["crm"] = crm_path

    gt_path = out_dir / "ground_truth.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "external_id", "true_entity_id", "is_confusable_distinct"]
        )
        writer.writeheader()
        for g in dataset.ground_truth:
            writer.writerow({
                "source": g.source,
                "external_id": g.external_id,
                "true_entity_id": g.true_entity_id,
                "is_confusable_distinct": int(g.is_confusable_distinct),
            })
    paths["ground_truth"] = gt_path

    return paths


def load_registry(path: str | Path) -> list[LoadedEntity]:
    out = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_address = ", ".join(
                p for p in [row["street_address"], row["city"], row["postal_code"]] if p
            )
            out.append(LoadedEntity(
                raw_entity=RawEntity(
                    source="registry",
                    external_id=row["reg_id"],
                    raw_name=row["legal_name"],
                    raw_address=raw_address,
                ),
                city=row["city"],
                postal_code=row["postal_code"],
            ))
    return out


def load_directory(path: str | Path) -> list[LoadedEntity]:
    out = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(LoadedEntity(
                raw_entity=RawEntity(
                    source="directory",
                    external_id=row["dir_id"],
                    raw_name=row["company_name"],
                    raw_address=row["full_address"],
                ),
                city="",  # not structured in this source -- normalize() must recover it
                postal_code="",
            ))
    return out


def load_crm(path: str | Path) -> list[LoadedEntity]:
    out = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_address = ", ".join(
                p for p in [row["billing_street"], row["billing_city"]] if p
            )
            out.append(LoadedEntity(
                raw_entity=RawEntity(
                    source="crm",
                    external_id=row["crm_id"],
                    raw_name=row["account_name"],
                    raw_address=raw_address,
                ),
                city=row["billing_city"],
                postal_code=row["billing_zip"],
            ))
    return out


LOADERS = {"registry": load_registry, "directory": load_directory, "crm": load_crm}


def load_all_sources(data_dir: str | Path) -> list[LoadedEntity]:
    data_dir = Path(data_dir)
    filenames = {
        "registry": "source_registry.csv",
        "directory": "source_directory.csv",
        "crm": "source_crm.csv",
    }
    entities: list[LoadedEntity] = []
    for source, filename in filenames.items():
        path = data_dir / filename
        if path.exists():
            entities.extend(LOADERS[source](path))
    return entities


def load_ground_truth(path: str | Path) -> list[GroundTruthRow]:
    out = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(GroundTruthRow(
                source=row["source"],
                external_id=row["external_id"],
                true_entity_id=row["true_entity_id"],
                is_confusable_distinct=bool(int(row["is_confusable_distinct"])),
            ))
    return out
