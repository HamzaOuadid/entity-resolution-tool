from entity_resolution.datagen import generate_dataset


def test_generate_dataset_is_deterministic_for_a_given_seed():
    d1 = generate_dataset(n_entities=40, seed=7)
    d2 = generate_dataset(n_entities=40, seed=7)
    names1 = [r.raw_name for r in d1.raw_entities]
    names2 = [r.raw_name for r in d2.raw_entities]
    assert names1 == names2


def test_generate_dataset_covers_all_three_sources():
    d = generate_dataset(n_entities=60, seed=1)
    sources = {r.source for r in d.raw_entities}
    assert sources == {"registry", "directory", "crm"}


def test_generate_dataset_has_multiple_appearances_per_entity():
    d = generate_dataset(n_entities=60, seed=1)
    counts: dict[str, int] = {}
    for g in d.ground_truth:
        counts[g.true_entity_id] = counts.get(g.true_entity_id, 0) + 1
    multi = [c for c in counts.values() if c >= 2]
    assert len(multi) > 0
    assert all(c >= 1 for c in counts.values())


def test_generate_dataset_includes_confusable_distinct_entities():
    d = generate_dataset(n_entities=100, seed=3, confusable_fraction=0.2)
    confusables = [g for g in d.ground_truth if g.is_confusable_distinct]
    assert len(confusables) > 0
    # confusable siblings must have DIFFERENT true_entity_id despite similar names
    ids = {g.true_entity_id for g in confusables}
    assert len(ids) >= 2


def test_generate_dataset_injects_missing_fields():
    d = generate_dataset(n_entities=150, seed=5, missing_field_rate=0.3)
    empty_address_parts = sum(1 for r in d.raw_entities if "" in r.raw_address.split(", "))
    # some rows should have a visibly shorter address because a field was dropped
    short_addresses = [r for r in d.raw_entities if len(r.raw_address.split(",")) < 3]
    assert len(short_addresses) > 0


def test_ground_truth_covers_every_raw_entity():
    d = generate_dataset(n_entities=50, seed=9)
    gt_keys = {(g.source, g.external_id) for g in d.ground_truth}
    raw_keys = {(r.source, r.external_id) for r in d.raw_entities}
    assert gt_keys == raw_keys


def test_external_ids_unique_within_source():
    d = generate_dataset(n_entities=80, seed=11)
    by_source: dict[str, set] = {}
    for r in d.raw_entities:
        by_source.setdefault(r.source, set())
        assert r.external_id not in by_source[r.source]
        by_source[r.source].add(r.external_id)
