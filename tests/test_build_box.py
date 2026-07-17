"""build_box: density/box arithmetic, determinism, and the manifest contract.

Layered so the parts that need no Packmol binary always run: the density
inversion and conformer determinism are pure, and the packing path is
exercised with Packmol stubbed. A real pack needs the binary and is covered
by the pipeline integration tests.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pymatgen")

from scilink.skills.structure_generation.condensed.build_box import (  # noqa: E402
    TOOL_SPEC, _molar_mass, _molecule_from_smiles, box_length_for_density,
    build_box,
)

WATER = {"name": "water", "smiles": "O", "count": 500}


# ── the density relation (pure) ──────────────────────────────────────

def test_box_length_reproduces_known_water_density():
    """500 waters at 1.0 g/cm^3 is a ~24.6 A cube — the textbook check."""
    edge = box_length_for_density([WATER], density=1.0)
    assert 24.0 < edge < 25.0


def test_box_length_inverts_the_density_relation_exactly():
    """Round-trip: the realized density of the returned edge is the target."""
    density = 1.05
    comps = [WATER, {"name": "eis", "smiles": "CCS(=O)(=O)C(C)C", "count": 40}]
    edge = box_length_for_density(comps, density=density)

    total_mass_g = sum(_molar_mass(c["smiles"]) * c["count"] for c in comps) / 6.02214076e23
    realized = total_mass_g / ((edge ** 3) * 1e-24)
    assert realized == pytest.approx(density, rel=1e-9)


def test_box_scales_with_count_not_with_species_identity():
    """Doubling every count doubles the volume (edge grows by 2**(1/3))."""
    single = box_length_for_density([WATER], density=1.0)
    doubled = box_length_for_density(
        [{**WATER, "count": WATER["count"] * 2}], density=1.0)
    assert doubled == pytest.approx(single * 2 ** (1 / 3), rel=1e-9)


def test_nonpositive_density_is_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="density must be positive"):
            box_length_for_density([WATER], density=bad)


# ── determinism (the whole point of the tool) ────────────────────────

def test_conformer_embedding_is_seeded():
    """Same seed -> identical coordinates; different seed -> different ones.

    Without a pinned seed every member of a 'same protocol' series would get
    different starting geometries, which is the failure this tool exists to
    prevent.
    """
    smiles = "CCS(=O)(=O)C(C)C"
    a = _molecule_from_smiles(smiles, "eis", conformer_seed=42)
    b = _molecule_from_smiles(smiles, "eis", conformer_seed=42)
    c = _molecule_from_smiles(smiles, "eis", conformer_seed=7)

    assert a.cart_coords.tolist() == b.cart_coords.tolist()
    assert a.cart_coords.tolist() != c.cart_coords.tolist()


def test_unparseable_smiles_names_the_component():
    with pytest.raises(ValueError, match="not-a-smiles"):
        _molecule_from_smiles("not-a-smiles", "bogus", conformer_seed=1)


# ── argument validation ──────────────────────────────────────────────

def test_no_components_rejected(tmp_path):
    with pytest.raises(ValueError, match="no components"):
        build_box([], density=1.0, working_dir=str(tmp_path))


def test_density_or_box_required(tmp_path):
    with pytest.raises(ValueError, match="either density .* or box"):
        build_box([WATER], working_dir=str(tmp_path))


def test_zero_count_component_rejected(tmp_path):
    """A count of 0 would desync the manifest from the packed coordinates."""
    with pytest.raises(ValueError, match="count"):
        build_box([{**WATER, "count": 0}], density=1.0, working_dir=str(tmp_path))


# ── the packing path, with Packmol stubbed ───────────────────────────

def _stub_packmol(tmp_path, captured):
    """Patch PackmolBoxGen to record its args and emit a 3-atom xyz."""
    class _Set:
        def write_input(self, d):
            captured["write_input_dir"] = d

        def run(self, path, timeout=30):
            captured["run_timeout"] = timeout
            Path(path, "packmol_out.xyz").write_text(
                "3\nstub\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0\n"
            )

    class _Gen:
        def __init__(self, tolerance=2.0, seed=1, **kw):
            captured["tolerance"] = tolerance
            captured["seed"] = seed

        def get_input_set(self, molecules, box=None):
            captured["molecules"] = molecules
            captured["box"] = box
            return _Set()

    return mock.patch(
        "pymatgen.io.packmol.PackmolBoxGen", _Gen)


def test_build_box_writes_manifest_in_component_order(tmp_path):
    """The manifest is a tool output, and its order is build_interchange's contract."""
    captured = {}
    comps = [
        {"name": "water", "smiles": "O", "count": 5},
        {"name": "eis", "smiles": "CCS(=O)(=O)C(C)C", "count": 2},
    ]
    with _stub_packmol(tmp_path, captured):
        out = build_box(comps, density=1.0, working_dir=str(tmp_path))

    manifest = json.loads(Path(out["manifest_file"]).read_text())
    assert [c["name"] for c in manifest["components"]] == ["water", "eis"]
    assert [c["count"] for c in manifest["components"]] == [5, 2]
    assert [c["smiles"] for c in manifest["components"]] == ["O", "CCS(=O)(=O)C(C)C"]


def test_build_box_forwards_seed_and_tolerance_to_packmol(tmp_path):
    """The reproducibility knobs must actually reach Packmol, not just exist."""
    captured = {}
    with _stub_packmol(tmp_path, captured):
        build_box([WATER], density=1.0, working_dir=str(tmp_path),
                  seed=1234, tolerance=2.5, timeout=99)

    assert captured["seed"] == 1234
    assert captured["tolerance"] == 2.5
    assert captured["run_timeout"] == 99


def test_build_box_passes_counts_through_to_packmol(tmp_path):
    captured = {}
    with _stub_packmol(tmp_path, captured):
        build_box([{**WATER, "count": 17}], density=1.0, working_dir=str(tmp_path))

    assert [m["number"] for m in captured["molecules"]] == [17]
    assert [m["name"] for m in captured["molecules"]] == ["water"]


def test_packing_region_is_inset_by_tolerance(tmp_path):
    """Molecules flush against a face would clash with their periodic images."""
    captured = {}
    with _stub_packmol(tmp_path, captured):
        out = build_box([WATER], box=30.0, working_dir=str(tmp_path), tolerance=2.0)

    lo, hi = captured["box"][0], captured["box"][3]
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(29.0)
    assert out["box"] == 30.0


def test_structure_gets_cell_and_pbc(tmp_path):
    """Packmol does not record a cell; build_interchange requires one."""
    from ase.io import read

    captured = {}
    with _stub_packmol(tmp_path, captured):
        out = build_box([WATER], box=30.0, working_dir=str(tmp_path))

    atoms = read(out["structure_file"])
    assert atoms.get_pbc().all()
    assert atoms.get_cell().lengths() == pytest.approx([30.0, 30.0, 30.0])


def test_missing_packmol_output_is_actionable(tmp_path):
    """A silent empty pack must fail loudly and point at the stdout log."""
    class _Set:
        def write_input(self, d): pass
        def run(self, path, timeout=30): pass  # writes nothing

    class _Gen:
        def __init__(self, **kw): pass
        def get_input_set(self, molecules, box=None): return _Set()

    with mock.patch("pymatgen.io.packmol.PackmolBoxGen", _Gen):
        with pytest.raises(RuntimeError, match="packmol.stdout"):
            build_box([WATER], density=1.0, working_dir=str(tmp_path))


# ── the TOOL_SPEC surface ────────────────────────────────────────────

def test_tool_spec_exposes_the_reproducibility_knobs():
    """Per the no-locked-knobs rule, seed/tolerance must be LLM-visible."""
    assert TOOL_SPEC.name == "build_box"
    assert "simulation" in TOOL_SPEC.agents
    for knob in ("seed", "conformer_seed", "tolerance", "density", "box"):
        assert knob in TOOL_SPEC.parameters, f"{knob} not exposed to the LLM"
    assert TOOL_SPEC.required == ["components"]


def test_tool_spec_is_resolvable_through_the_registry():
    """A TOOL_SPEC only counts if get_tool_function can find it."""
    from scilink.skills._shared._registry import get_tool_function

    fn = get_tool_function("build_box", active_skills=["condensed"])
    assert fn is build_box
