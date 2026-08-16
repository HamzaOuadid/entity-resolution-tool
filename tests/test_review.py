from entity_resolution.models import CandidatePair, NormalizedEntity
from entity_resolution.review import explain_pair, export_review_queue_csv, format_explanation


def _entity(id_, name, city, postal, address):
    return NormalizedEntity(
        id=id_, raw_entity_id=id_, normalized_name=name, normalized_address=address,
        normalized_city=city, normalized_postal_code=postal,
        name_core_tokens=tuple(sorted(name.split())), source="registry",
    )


def test_explain_pair_flags_name_agreement_and_postal_disagreement():
    a = _entity(1, "acme industrial holdings bv", "amsterdam", "1017ab", "kerkstraat 12")
    b = _entity(2, "acme industrial holdings bv", "rotterdam", "3011ce", "kerkstraat 12")
    pair = CandidatePair(id=1, entity_a_id=1, entity_b_id=2, blocking_key="k", similarity_score=0.7)
    exp = explain_pair(pair, a, b)
    assert "name (word-order-independent)" in exp.agreements
    assert "postal code" in exp.disagreements


def test_explain_pair_flags_missing_data_as_uncertain_not_disagreement():
    a = _entity(1, "acme industrial", "", "", "")
    b = _entity(2, "acme industrial", "denver", "80202", "500 oak street")
    pair = CandidatePair(id=1, entity_a_id=1, entity_b_id=2, blocking_key="k", similarity_score=0.7)
    exp = explain_pair(pair, a, b)
    assert "street address" in exp.uncertain
    assert "postal code" in exp.uncertain
    assert "city" in exp.uncertain
    assert "street address" not in exp.disagreements


def test_format_explanation_includes_pair_id_and_score():
    a = _entity(1, "acme industrial", "denver", "80202", "500 oak street")
    b = _entity(2, "acme industrial group", "denver", "80202", "500 oak street")
    pair = CandidatePair(id=42, entity_a_id=1, entity_b_id=2, blocking_key="k", similarity_score=0.7)
    exp = explain_pair(pair, a, b)
    text = format_explanation(exp)
    assert "pair_id=42" in text
    assert "0.700" in text


def test_export_review_queue_csv_writes_expected_rows(tmp_path):
    a = _entity(1, "acme industrial", "denver", "80202", "500 oak street")
    b = _entity(2, "acme industrial group", "denver", "80202", "500 oak street")
    pair = CandidatePair(id=1, entity_a_id=1, entity_b_id=2, blocking_key="k", similarity_score=0.7)
    exp = explain_pair(pair, a, b)
    out_path = export_review_queue_csv([exp], tmp_path / "review.csv")
    assert out_path.exists()

    import csv
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "1"
    assert rows[0]["reviewer_decision"] == ""
