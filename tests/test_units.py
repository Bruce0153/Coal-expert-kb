from coal_kb.schema.units import atm_to_mpa, bar_to_mpa, c_to_k, mol_percent_to_ppmv, ppmv_to_mol_percent


def test_temp():
    assert abs(c_to_k(0) - 273.15) < 1e-6


def test_pressure():
    assert abs(bar_to_mpa(10) - 1.0) < 1e-9
    assert abs(atm_to_mpa(1) - 0.101325) < 1e-9


def test_mol_ppm():
    assert abs(mol_percent_to_ppmv(1.0) - 10000.0) < 1e-9
    assert abs(ppmv_to_mol_percent(10000.0) - 1.0) < 1e-9
