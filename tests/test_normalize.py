from entity_resolution.models import RawEntity
from entity_resolution.normalize import (
    extract_postal_code,
    normalize,
    normalize_address_text,
    normalize_city,
    normalize_name,
    normalize_postal_code,
)


def test_legal_suffix_variants_normalize_to_same_token():
    variants = ["ACME Industrial Holdings B.V.", "Acme Industrial Holdings BV", "ACME INDUSTRIAL HOLDINGS B.V"]
    results = [normalize_name(v) for v in variants]
    suffixes = {r[1] for r in results}
    assert suffixes == {"bv"}


def test_name_core_tokens_ignore_word_order():
    a = normalize_name("Northgate Bluewater Logistics Inc.")
    b = normalize_name("Bluewater Northgate Logistics Inc")
    assert a[2] == b[2]  # sorted core tokens identical regardless of order


def test_abbreviation_expansion_in_name():
    normalized, _, _ = normalize_name("Summit Mfg Co.")
    assert "manufacturing" in normalized


def test_ampersand_and_punctuation_are_normalized():
    a, _, _ = normalize_name("Smith & Sons, Inc.")
    b, _, _ = normalize_name("Smith and Sons Inc")
    assert a == b


def test_address_abbreviations_expand():
    result = normalize_address_text("123 Maple St")
    assert "street" in result
    assert "st" not in result.split()


def test_extract_postal_code_from_free_text_us():
    assert extract_postal_code("123 Maple St, Denver, CO 80202") == "80202"


def test_extract_postal_code_from_free_text_eu():
    assert extract_postal_code("Kerkstraat 12, 1017 AB Amsterdam") == "1017AB"


def test_extract_postal_code_missing_returns_empty():
    assert extract_postal_code("123 Maple St, Denver, CO") == ""


def test_normalize_postal_code_strips_whitespace_and_uppercases():
    assert normalize_postal_code("1017 ab") == "1017AB"


def test_normalize_city_handles_saint_abbreviation():
    assert normalize_city("St. Louis") == "saint louis"


def test_normalize_end_to_end_recovers_postal_from_free_text():
    entity = RawEntity(
        id=1, source="directory", external_id="DIR-1",
        raw_name="Acme Industrial Holdings BV",
        raw_address="Kerkstraat 12, Amsterdam, 1017 AB",
    )
    result = normalize(entity)
    assert result.normalized_postal_code == "1017AB"
    assert result.legal_suffix == "bv"
    assert result.raw_entity_id == 1


def test_normalize_uses_structured_fields_when_given():
    entity = RawEntity(
        id=2, source="registry", external_id="REG-1",
        raw_name="Acme Industrial Holdings B.V.",
        raw_address="Kerkstraat 12",
    )
    result = normalize(entity, city="Amsterdam", postal_code="1017 AB")
    assert result.normalized_city == "amsterdam"
    assert result.normalized_postal_code == "1017AB"


def test_missing_address_fields_do_not_raise():
    entity = RawEntity(id=3, source="crm", external_id="CRM-1", raw_name="Vantage Foods LLC", raw_address="")
    result = normalize(entity)
    assert result.normalized_address == ""
    assert result.normalized_postal_code == ""
