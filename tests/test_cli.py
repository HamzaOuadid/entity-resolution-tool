from typer.testing import CliRunner

from entity_resolution.cli import app

runner = CliRunner()


def test_generate_data_writes_expected_files(tmp_path):
    out_dir = tmp_path / "data"
    result = runner.invoke(app, ["generate-data", "--out", str(out_dir), "--n-entities", "60", "--seed", "5"])
    assert result.exit_code == 0, result.output
    assert (out_dir / "source_registry.csv").exists()
    assert (out_dir / "source_directory.csv").exists()
    assert (out_dir / "source_crm.csv").exists()
    assert (out_dir / "ground_truth.csv").exists()


def test_run_all_prints_report_and_persists_db(tmp_path):
    out_dir = tmp_path / "data"
    db_path = tmp_path / "er.db"
    report_path = tmp_path / "report.json"
    gen = runner.invoke(app, ["generate-data", "--out", str(out_dir), "--n-entities", "80", "--seed", "5"])
    assert gen.exit_code == 0, gen.output

    result = runner.invoke(app, [
        "run-all", "--data-dir", str(out_dir), "--db", str(db_path), "--report-out", str(report_path),
    ])
    assert result.exit_code == 0, result.output
    assert "Blocking recall" in result.output
    assert db_path.exists()
    assert report_path.exists()

    import json
    metrics = json.loads(report_path.read_text())
    assert metrics["n_entities"] > 0
    assert 0.0 <= metrics["full_dataset_precision"] <= 1.0


def test_review_queue_shows_feature_breakdown(tmp_path):
    out_dir = tmp_path / "data"
    db_path = tmp_path / "er.db"
    runner.invoke(app, ["generate-data", "--out", str(out_dir), "--n-entities", "300", "--seed", "42"])
    runner.invoke(app, ["run-all", "--data-dir", str(out_dir), "--db", str(db_path)])

    result = runner.invoke(app, ["review-queue", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "pairs awaiting review" in result.output
    assert "review band" in result.output


def test_export_review_queue_writes_csv(tmp_path):
    out_dir = tmp_path / "data"
    db_path = tmp_path / "er.db"
    csv_path = tmp_path / "review.csv"
    runner.invoke(app, ["generate-data", "--out", str(out_dir), "--n-entities", "300", "--seed", "42"])
    runner.invoke(app, ["run-all", "--data-dir", str(out_dir), "--db", str(db_path)])

    result = runner.invoke(app, ["export-review-queue", "--db", str(db_path), "--out", str(csv_path)])
    assert result.exit_code == 0, result.output
    assert csv_path.exists()

    import csv
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "reviewer_decision" in rows[0]


def test_submit_review_records_decision_and_closes_the_loop(tmp_path):
    out_dir = tmp_path / "data"
    db_path = tmp_path / "er.db"
    runner.invoke(app, ["generate-data", "--out", str(out_dir), "--n-entities", "300", "--seed", "42"])
    runner.invoke(app, ["run-all", "--data-dir", str(out_dir), "--db", str(db_path)])

    from entity_resolution.db import EntityResolutionDB
    with EntityResolutionDB(db_path) as db:
        review_decisions = [d for d in db.get_decisions() if d.decision == "review"]
    assert review_decisions, "expected at least one review-band pair"
    pair_id = review_decisions[0].pair_id

    result = runner.invoke(app, ["submit-review", str(pair_id), "match", "--reviewer", "hamza", "--db", str(db_path)])
    assert result.exit_code == 0, result.output

    with EntityResolutionDB(db_path) as db:
        updated = [d for d in db.get_decisions() if d.pair_id == pair_id][0]
    assert updated.decision == "match"
    assert updated.reviewer == "hamza"

    # the pair must have left the review queue now that it's decided
    result2 = runner.invoke(app, ["review-queue", "--db", str(db_path)])
    assert f"pair_id={pair_id} " not in result2.output


def test_submit_review_on_unknown_pair_warns_but_still_records(tmp_path):
    out_dir = tmp_path / "data"
    db_path = tmp_path / "er.db"
    runner.invoke(app, ["generate-data", "--out", str(out_dir), "--n-entities", "80", "--seed", "5"])
    runner.invoke(app, ["run-all", "--data-dir", str(out_dir), "--db", str(db_path)])

    result = runner.invoke(app, ["submit-review", "999999", "match", "--reviewer", "hamza", "--db", str(db_path)])
    assert result.exit_code == 0
    assert "warning" in result.output.lower()
