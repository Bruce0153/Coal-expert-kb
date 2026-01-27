import pytest

from coal_kb.schema.validators import validate_pressure_mpa, validate_temperature_k


def test_validate_temperature_ok():
    validate_temperature_k(1200.0)


def test_validate_temperature_bad():
    with pytest.raises(ValueError):
        validate_temperature_k(5000.0)


def test_validate_pressure_ok():
    validate_pressure_mpa(2.0)


def test_validate_pressure_bad():
    with pytest.raises(ValueError):
        validate_pressure_mpa(-1.0)
