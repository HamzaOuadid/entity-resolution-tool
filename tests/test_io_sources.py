from entity_resolution.datagen import generate_dataset
from entity_resolution.io_sources import load_all_sources, load_ground_truth, write_dataset


def test_write_and_reload_dataset_round_trips_row_counts(tmp_path):
    dataset = generate_dataset(n_entities=60, seed=17)
    paths = write_dataset(dataset, tmp_path)
    assert paths["registry"].exists()
    assert paths["directory"].exists()
    assert paths["crm"].exists()
    assert paths["ground_truth"].exists()

    loaded = load_all_sources(tmp_path)
    assert len(loaded) == len(dataset.raw_entities)

    gt = load_ground_truth(paths["ground_truth"])
    assert len(gt) == len(dataset.ground_truth)


def test_directory_source_has_no_structured_city_but_address_recoverable(tmp_path):
    dataset = generate_dataset(n_entities=40, seed=19)
    paths = write_dataset(dataset, tmp_path)
    loaded = load_all_sources(tmp_path)
    directory_rows = [e for e in loaded if e.raw_entity.source == "directory"]
    assert directory_rows
    assert all(e.city == "" for e in directory_rows)
    assert all(len(e.raw_entity.raw_address) > 0 for e in directory_rows if e.raw_entity.raw_address)


def test_registry_source_has_structured_city():
    from entity_resolution.datagen import generate_dataset
    from entity_resolution.io_sources import write_dataset, load_all_sources
    import tempfile, pathlib
    dataset = generate_dataset(n_entities=40, seed=23)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(dataset, d)
        loaded = load_all_sources(d)
        registry_rows = [e for e in loaded if e.raw_entity.source == "registry"]
        with_city = [e for e in registry_rows if e.city]
        assert len(with_city) > 0
