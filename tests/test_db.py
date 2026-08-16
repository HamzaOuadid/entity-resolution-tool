from entity_resolution.db import EntityResolutionDB
from entity_resolution.models import CandidatePair, NormalizedEntity, RawEntity, ResolutionDecision


def test_insert_and_get_raw_entity():
    with EntityResolutionDB(":memory:") as db:
        rid = db.insert_raw_entity(RawEntity(source="registry", external_id="R1", raw_name="Acme", raw_address="1 Main St"))
        assert rid > 0
        rows = db.get_raw_entities()
        assert len(rows) == 1
        assert rows[0].raw_name == "Acme"


def test_insert_raw_entity_is_idempotent_on_source_and_external_id():
    with EntityResolutionDB(":memory:") as db:
        rid1 = db.insert_raw_entity(RawEntity(source="registry", external_id="R1", raw_name="Acme", raw_address="A"))
        rid2 = db.insert_raw_entity(RawEntity(source="registry", external_id="R1", raw_name="Acme Dup Insert", raw_address="A"))
        assert rid1 == rid2
        assert len(db.get_raw_entities()) == 1


def test_normalized_entity_roundtrip_preserves_tokens():
    with EntityResolutionDB(":memory:") as db:
        rid = db.insert_raw_entity(RawEntity(source="crm", external_id="C1", raw_name="Acme Industrial", raw_address=""))
        db.insert_normalized_entity(NormalizedEntity(
            raw_entity_id=rid, normalized_name="acme industrial", normalized_address="",
            normalized_city="denver", normalized_postal_code="80202", legal_suffix="",
            name_core_tokens=("acme", "industrial"), source="crm",
        ))
        rows = db.get_normalized_entities()
        assert rows[0].name_core_tokens == ("acme", "industrial")


def test_candidate_pair_ordering_is_canonicalized():
    with EntityResolutionDB(":memory:") as db:
        pid = db.insert_candidate_pair(CandidatePair(entity_a_id=5, entity_b_id=2, blocking_key="k"))
        pairs = db.get_candidate_pairs()
        assert pairs[0].entity_a_id == 2
        assert pairs[0].entity_b_id == 5
        assert pairs[0].id == pid


def test_decision_upsert_updates_existing_row():
    with EntityResolutionDB(":memory:") as db:
        pid = db.insert_candidate_pair(CandidatePair(entity_a_id=1, entity_b_id=2, blocking_key="k"))
        db.upsert_decision(ResolutionDecision(pair_id=pid, decision="review"))
        db.upsert_decision(ResolutionDecision(pair_id=pid, decision="match", reviewer="hamza"))
        decisions = db.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].decision == "match"
        assert decisions[0].reviewer == "hamza"


def test_reset_clears_all_tables():
    with EntityResolutionDB(":memory:") as db:
        db.insert_raw_entity(RawEntity(source="crm", external_id="C1", raw_name="Acme", raw_address=""))
        db.reset()
        assert db.get_raw_entities() == []
