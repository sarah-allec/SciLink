"""ECC charge scaling: scale ions, leave neutral species intact, keep neutrality.

Pure arithmetic on per-molecule charge lists — no force-field toolkit.
"""

import pytest

from scilink.skills._shared._charge_scaling import (
    DEFAULT_ECC_FACTOR, is_ionic, scale_ionic_charges)


def test_ions_scaled_neutral_untouched():
    # A neutral box: Zn2+ , two OTf- (as net -1 each), and water (neutral).
    zn = [2.0]
    otf_a, otf_b = [-1.0], [-1.0]
    water = [0.4, -0.8, 0.4]                       # sums to 0
    out = scale_ionic_charges([zn, otf_a, otf_b, water], 0.75)
    assert out[0] == [1.5]                         # Zn2+ -> +1.5
    assert out[1] == [-0.75] and out[2] == [-0.75] # OTf- -> -0.75
    assert out[3] == [0.4, -0.8, 0.4]              # water unchanged (full dipole)


def test_total_charge_preserved_for_neutral_box():
    zn, otf_a, otf_b = [2.0], [-1.0], [-1.0]
    water = [0.4, -0.8, 0.4]
    out = scale_ionic_charges([zn, otf_a, otf_b, water], 0.75)
    assert abs(sum(sum(m) for m in out)) < 1e-9    # still neutral


def test_polyatomic_ion_scales_every_atom():
    # A polyatomic anion with distributed partial charges summing to -1.
    otf = [0.5, -0.5, -0.4, -0.6]                  # sum = -1.0
    out = scale_ionic_charges([otf], 0.8)[0]
    assert out == pytest.approx([0.4, -0.4, -0.32, -0.48])
    assert sum(out) == pytest.approx(-0.8)         # net -1 -> -0.8


def test_inputs_not_mutated():
    zn = [2.0]
    scale_ionic_charges([zn], 0.5)
    assert zn == [2.0]


def test_default_factor_is_ecc_value():
    assert abs(DEFAULT_ECC_FACTOR - 0.75) < 1e-9


def test_is_ionic_tolerance():
    assert is_ionic([1.0, -0.5]) is True
    assert is_ionic([0.4, -0.8, 0.4]) is False     # rounds to neutral
