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
