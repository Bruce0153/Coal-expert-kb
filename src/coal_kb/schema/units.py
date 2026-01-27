from __future__ import annotations


def c_to_k(c: float) -> float:
    return c + 273.15


def k_to_c(k: float) -> float:
    return k - 273.15


def bar_to_mpa(bar: float) -> float:
    return bar * 0.1


def atm_to_mpa(atm: float) -> float:
    return atm * 0.101325


def mol_percent_to_ppmv(mol_percent: float) -> float:
    # 1 mol% = 10,000 ppmv
    return mol_percent * 10000.0


def ppmv_to_mol_percent(ppmv: float) -> float:
    return ppmv / 10000.0
