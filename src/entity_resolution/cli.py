"""Typer CLI. Two ways to use the pipeline:

  * `run-all` / `evaluate` : in-memory, via pipeline.run_pipeline() -- fast,
    what the README demo and CI use to get real precision/recall numbers.
  * `load` / `normalize-db` / `block-db` / `score-db` / `resolve-db` :
    step-by-step, persisting every stage to the SQLite data model (matches
    the spec's raw_entities/normalized_entities/candidate_pairs/
    resolution_decisions tables), so `review-queue` / `submit-review` have
    something durable to work against.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .blocking import block as block_fn
from .datagen import generate_dataset
from .db import EntityResolutionDB
from .evaluation import evaluate_against_hand_labels, hand_label_sample as hand_label_sample_fn, true_match_pairs, blocking_recall as blocking_recall_fn, pairwise_prf
from .io_sources import load_all_sources, load_ground_truth, write_dataset
from .models import CandidatePair, NormalizedEntity, ResolutionDecision
from .normalize import normalize as normalize_fn
from .pipeline import run_pipeline
from .resolve import ResolveThresholds, resolve as resolve_fn
from .review import explain_pair, export_review_queue_csv, format_explanation
from .scoring import HeuristicScorer, score_all

app = typer.Typer(add_completion=False, help="Entity resolution / dedup pipeline for messy multi-source data.")


@app.command()
def generate_data(
    out: Path = typer.Option(Path("data"), help="Output directory for the source CSVs + ground truth"),
    n_entities: int = typer.Option(350, help="Number of true underlying entities to generate"),
    seed: int = typer.Option(42),
    confusable_fraction: float = typer.Option(0.12, help="Fraction of entities that are confusable franchise-style siblings"),
    missing_field_rate: float = typer.Option(0.12),
):
    """Generate the messy multi-source synthetic dataset with known ground truth."""
    dataset = generate_dataset(
        n_entities=n_entities, seed=seed,
        confusable_fraction=confusable_fraction, missing_field_rate=missing_field_rate,
    )
    paths = write_dataset(dataset, out, seed=seed)
    typer.echo(f"Wrote {len(dataset.raw_entities)} raw rows across 3 sources to {out}/")
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")


@app.command()
def run_all(
    data_dir: Path = typer.Option(Path("data")),
    match_threshold: float = typer.Option(0.83),
    no_match_threshold: float = typer.Option(0.55),
    sample_size: int = typer.Option(300),
    db: Path = typer.Option(Path("entity_resolution.db")),
    report_out: Path = typer.Option(None, help="Optional path to write the metrics report as JSON"),
):
    """Run the full pipeline end-to-end against a CSV dataset on disk, print
    a metrics report, and persist every stage to SQLite."""
    loaded = load_all_sources(data_dir)
    gt_path = data_dir / "ground_truth.csv"
    if not gt_path.exists():
        typer.echo(f"No ground_truth.csv found in {data_dir}; run generate-data first.", err=True)
        raise typer.Exit(1)
    ground_truth = load_ground_truth(gt_path)

    from .datagen import GeneratedDataset
    raw_entities = [le.raw_entity for le in loaded]
    dataset = GeneratedDataset(raw_entities=raw_entities, ground_truth=ground_truth)

    thresholds = ResolveThresholds(match_threshold=match_threshold, no_match_threshold=no_match_threshold)
    result = run_pipeline(dataset, thresholds=thresholds, sample_size=sample_size)

    _print_report(result.metrics)

    if report_out:
        report_out.write_text(json.dumps(result.metrics, indent=2))
        typer.echo(f"\nWrote metrics to {report_out}")

    _persist_to_db(db, loaded, result)
    typer.echo(f"Persisted pipeline state to {db}")


def _print_report(metrics: dict) -> None:
    typer.echo("=== Entity Resolution Report ===")
    typer.echo(f"Entities:                {metrics['n_entities']}")
    typer.echo(f"Candidate pairs (post-blocking): {metrics['n_candidate_pairs']}")
    typer.echo(f"True match pairs (ground truth):  {metrics['n_true_match_pairs']}")
    typer.echo("")
    typer.echo(f"Blocking recall:          {metrics['blocking_recall']:.3f}  ({metrics['blocking_missed']} true matches missed by blocking)")
    typer.echo("")
    typer.echo("-- Full-dataset match quality (all candidate pairs vs. ground truth) --")
    typer.echo(f"Precision: {metrics['full_dataset_precision']:.3f}")
    typer.echo(f"Recall:    {metrics['full_dataset_recall']:.3f}")
    typer.echo(f"F1:        {metrics['full_dataset_f1']:.3f}")
    typer.echo("")
    if "hand_label_precision" in metrics:
        typer.echo(f"-- Hand-labelled sample (n={metrics['hand_label_sample_size']}) --")
        typer.echo(f"Precision: {metrics['hand_label_precision']:.3f}  (tp={metrics['hand_label_tp']} fp={metrics['hand_label_fp']})")
        typer.echo(f"Recall:    {metrics['hand_label_recall']:.3f}  (fn={metrics['hand_label_fn']})")
        typer.echo(f"F1:        {metrics['hand_label_f1']:.3f}")
        typer.echo("")
    typer.echo(f"Decisions: {metrics['n_matches']} match / {metrics['n_no_matches']} no_match / {metrics['n_review_queue']} review")
    typer.echo(f"Clusters: {metrics['n_clusters']} total, {metrics['n_flagged_conflict_clusters']} flagged for conflicting evidence")


def _persist_to_db(db_path: Path, loaded, result) -> None:
    with EntityResolutionDB(db_path) as db:
        db.reset()
        raw_id_map = {}
        for le in loaded:
            rid = db.insert_raw_entity(le.raw_entity)
            raw_id_map[(le.raw_entity.source, le.raw_entity.external_id)] = rid

        # normalized_entities keyed by our in-memory sequential id (1..n);
        # re-derive the persisted raw_entity_id via the same (source, external_id).
        # Mapping from pipeline entity id -> raw (source, external_id) relies on
        # the order entities were created (see pipeline.normalize_dataset).
        norm_id_map = {}
        for i, le in enumerate(loaded):
            pipeline_id = i + 1
            entity = result.entities_by_id[pipeline_id]
            rid = raw_id_map[(le.raw_entity.source, le.raw_entity.external_id)]
            db_entity = NormalizedEntity(
                raw_entity_id=rid,
                normalized_name=entity.normalized_name,
                normalized_address=entity.normalized_address,
                normalized_city=entity.normalized_city,
                normalized_postal_code=entity.normalized_postal_code,
                legal_suffix=entity.legal_suffix,
                name_core_tokens=entity.name_core_tokens,
                source=entity.source,
            )
            db_id = db.insert_normalized_entity(db_entity)
            norm_id_map[pipeline_id] = db_id

        pair_id_map = {}
        for p in result.candidate_pairs:
            db_pair = CandidatePair(
                entity_a_id=norm_id_map[p.entity_a_id],
                entity_b_id=norm_id_map[p.entity_b_id],
                blocking_key=p.blocking_key,
                similarity_score=p.similarity_score,
            )
            db_id = db.insert_candidate_pair(db_pair)
            db.update_pair_score(db_id, p.similarity_score)
            pair_id_map[p.id] = db_id

        for p in result.resolve_result.matches:
            db.upsert_decision(ResolutionDecision(pair_id=pair_id_map[p.id], decision="match"))
        for p in result.resolve_result.no_matches:
            db.upsert_decision(ResolutionDecision(pair_id=pair_id_map[p.id], decision="no_match"))
        for p in result.resolve_result.review_queue:
            db.upsert_decision(ResolutionDecision(pair_id=pair_id_map[p.id], decision="review"))


def _load_review_queue(db_path: Path):
    with EntityResolutionDB(db_path) as conn:
        entities = {e.id: e for e in conn.get_normalized_entities()}
        pairs = {p.id: p for p in conn.get_candidate_pairs()}
        decisions = [d for d in conn.get_decisions() if d.decision == "review"]
    explanations = []
    for d in decisions:
        pair = pairs[d.pair_id]
        a, b = entities[pair.entity_a_id], entities[pair.entity_b_id]
        explanations.append(explain_pair(pair, a, b))
    return explanations


@app.command()
def review_queue(
    db: Path = typer.Option(Path("entity_resolution.db")),
    limit: int = typer.Option(20),
):
    """List ambiguous pairs routed to the review queue -- the spec's
    reviewer story: this band is surfaced explicitly, never silently
    auto-merged or silently dropped. Each pair is shown with a feature
    breakdown (what agrees/disagrees/is missing) so a reviewer can decide
    quickly instead of re-deriving the ambiguity from raw fields."""
    explanations = _load_review_queue(db)
    if not explanations:
        typer.echo("Review queue is empty.")
        return
    typer.echo(f"{len(explanations)} pairs awaiting review (showing up to {limit}):\n")
    for exp in explanations[:limit]:
        typer.echo(format_explanation(exp))
        typer.echo("")


@app.command()
def export_review_queue(
    db: Path = typer.Option(Path("entity_resolution.db")),
    out: Path = typer.Option(Path("review_queue.csv")),
):
    """Export the full review queue to a CSV a human reviewer can work
    through offline (e.g. in a spreadsheet), with a blank
    reviewer_decision column to fill in."""
    explanations = _load_review_queue(db)
    if not explanations:
        typer.echo("Review queue is empty; nothing to export.")
        raise typer.Exit(0)
    path = export_review_queue_csv(explanations, out)
    typer.echo(f"Exported {len(explanations)} review-queue pairs to {path}")


@app.command()
def submit_review(
    pair_id: int,
    decision: str = typer.Argument(..., help="'match' or 'no_match'"),
    reviewer: str = typer.Option(..., help="Reviewer name/id"),
    db: Path = typer.Option(Path("entity_resolution.db")),
):
    """Record a human reviewer's decision for a pair in the review queue.
    This is the other half of the reviewer story: the ambiguous band isn't
    a dead end -- a decision made here is durably recorded against the pair
    (resolution_decisions.reviewer / .decision), closing the loop."""
    if decision not in ("match", "no_match"):
        typer.echo("decision must be 'match' or 'no_match'", err=True)
        raise typer.Exit(1)
    with EntityResolutionDB(db) as conn:
        existing = {d.pair_id: d for d in conn.get_decisions()}
        if pair_id not in existing or existing[pair_id].decision != "review":
            typer.echo(
                f"warning: pair {pair_id} was not in the review queue (decision="
                f"{existing.get(pair_id).decision if pair_id in existing else 'unknown'}); recording anyway",
                err=True,
            )
        conn.upsert_decision(ResolutionDecision(pair_id=pair_id, decision=decision, reviewer=reviewer))
    typer.echo(f"Recorded {decision} for pair {pair_id} by {reviewer}")


if __name__ == "__main__":
    app()
