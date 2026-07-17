"""Pack a periodic box of component molecules at a target density.

The structure-side mirror of the force-field backend's ``build_interchange``:
given the per-component chemistry (SMILES + counts) it packs a periodic box
with Packmol and writes engine-neutral coordinates plus the components
manifest that ``build_interchange`` consumes. Same ``{name, smiles, count}``
vocabulary, one stage earlier, so ``build_box -> build_interchange`` is a
straight hand-off in coordinate order.

Packing is **reproducible**: both sources of randomness are seeded — the
RDKit conformer embedding that turns each SMILES into 3D coordinates, and
Packmol's own placement. Two calls with the same arguments produce the same
box, and two calls differing only in ``components`` counts produce boxes that
differ only by composition. That is what makes a composition series a
controlled comparison rather than N independent guesses.

Heavy deps (rdkit, pymatgen) are imported lazily; Packmol itself is an
external binary.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec

_PACKMOL_HINT = (
    "build_box needs the Packmol binary on PATH and pymatgen/rdkit:  "
    "conda install -c conda-forge packmol pymatgen rdkit"
)

# CODATA. Used for the density <-> box-volume relation.
_AVOGADRO = 6.02214076e23
# 1 cm^3 = 1e24 A^3.
_CM3_PER_A3 = 1e-24


def _molecule_from_smiles(smiles: str, name: str, conformer_seed: int):
    """Embed a SMILES into a single 3D pymatgen Molecule, deterministically."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from pymatgen.core import Molecule
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(f"{_PACKMOL_HINT}\n(original error: {e})") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"build_box: RDKit could not parse SMILES {smiles!r} "
                         f"for component {name!r}.")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    # Pin the embedding RNG: without this each call yields a different
    # conformer, so a "same protocol" series would silently differ per member.
    params.randomSeed = conformer_seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(
            f"build_box: RDKit could not embed a 3D conformer for {name!r} "
            f"(SMILES {smiles!r}). Supply an explicit coordinates file instead."
        )
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    species = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return Molecule(species, coords)


def _molar_mass(smiles: str) -> float:
    """Molar mass (g/mol) of one molecule of ``smiles``, including hydrogens."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return Descriptors.MolWt(mol)


def box_length_for_density(components: List[Dict[str, Any]],
                           density: float) -> float:
    """Cubic box edge (A) giving ``density`` (g/cm^3) for ``components``.

    Inverts the relation the condensed skill states in prose,
    rho = (sum_i N_i * M_i) / (N_A * V), for a cubic cell. Pure and
    unit-testable: no packing, no I/O.

    Args:
        components: ``[{name, smiles, count}]``.
        density: Target mass density in g/cm^3.

    Returns:
        The cubic box edge length in Angstrom.
    """
    if density <= 0:
        raise ValueError(f"build_box: density must be positive, got {density}.")
    total_mass_g = sum(
        _molar_mass(c["smiles"]) * int(c["count"]) for c in components
    ) / _AVOGADRO
    volume_cm3 = total_mass_g / density
    volume_a3 = volume_cm3 / _CM3_PER_A3
    return float(volume_a3 ** (1.0 / 3.0))


def build_box(
    components: List[Dict[str, Any]],
    density: Optional[float] = None,
    box: Optional[float] = None,
    working_dir: str = ".",
    output_file: str = "structure.extxyz",
    seed: int = 1,
    conformer_seed: int = 42,
    tolerance: float = 2.0,
    timeout: float = 300,
) -> Dict[str, Any]:
    """Pack a periodic box of molecules at a target density, reproducibly.

    Embeds each component's SMILES into a 3D conformer, sizes a cubic cell to
    ``density`` (or uses ``box``), packs the requested counts with Packmol, and
    writes extended-XYZ coordinates with cell and PBC set, alongside the
    ``components.json`` manifest in coordinate order.

    Args:
        components: ``[{name, smiles, count}]``. Written to the manifest in
            this order, which is the order the coordinates are packed in and
            the order ``build_interchange`` requires.
        density: Target density in g/cm^3 (e.g. 1.0 for aqueous). Ignored when
            ``box`` is given; one of the two is required.
        box: Cubic box edge in Angstrom. Overrides ``density``.
        working_dir: Directory for the coordinates, manifest, and Packmol
            scratch files.
        output_file: Coordinates filename written under ``working_dir``.
        seed: Packmol placement seed. Change it to draw an independent packing
            of the same composition (e.g. to estimate run-to-run spread); keep
            it fixed across a composition series so members differ only in
            composition.
        conformer_seed: RDKit embedding seed for the per-molecule 3D
            conformers. Same rule as ``seed``.
        tolerance: Packmol minimum inter-atomic separation in Angstrom. RAISE
            toward ~2.5 if the pack yields close contacts a minimizer cannot
            relax; LOWER toward ~1.5 only for dense systems Packmol otherwise
            fails to fit.
        timeout: Seconds allowed for the Packmol run. RAISE for large or dense
            boxes, where placement takes longer.

    Returns:
        dict with ``structure_file``, ``manifest_file``, ``box`` (edge in A),
        ``density`` (realized g/cm^3), ``n_atoms``, and ``n_molecules``.
    """
    try:
        from ase.io import read, write
        from pymatgen.io.packmol import PackmolBoxGen
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(f"{_PACKMOL_HINT}\n(original error: {e})") from e

    if not components:
        raise ValueError("build_box: no components supplied")
    if density is None and box is None:
        raise ValueError("build_box: supply either density (g/cm^3) or box (A)")
    for comp in components:
        if int(comp.get("count", 0)) < 1:
            raise ValueError(
                f"build_box: component {comp.get('name')!r} has count "
                f"{comp.get('count')!r}; every component needs count >= 1."
            )

    os.makedirs(working_dir, exist_ok=True)
    edge = float(box) if box is not None else box_length_for_density(
        components, density)

    molecules = [
        {
            "name": comp["name"],
            "number": int(comp["count"]),
            "coords": _molecule_from_smiles(
                comp["smiles"], comp["name"], conformer_seed),
        }
        for comp in components
    ]

    # Inset the packing region by the tolerance so molecules are not placed
    # flush against the cell faces, where their periodic images would clash.
    lo, hi = tolerance / 2.0, edge - tolerance / 2.0
    packmol_set = PackmolBoxGen(tolerance=tolerance, seed=seed).get_input_set(
        molecules=molecules, box=[lo, lo, lo, hi, hi, hi]
    )
    packmol_set.write_input(working_dir)
    packmol_set.run(working_dir, timeout=timeout)

    # PackmolSet exposes no structure: read the packed coordinates back off
    # disk, then attach the periodic cell Packmol itself does not record.
    packed_xyz = os.path.join(working_dir, "packmol_out.xyz")
    if not os.path.exists(packed_xyz):
        raise RuntimeError(
            f"build_box: Packmol produced no output at {packed_xyz}. Check "
            f"{os.path.join(working_dir, 'packmol.stdout')} — a dense box may "
            "need a larger timeout or a lower tolerance."
        )
    atoms = read(packed_xyz)
    atoms.set_cell([edge, edge, edge])
    atoms.set_pbc(True)

    structure_file = os.path.join(working_dir, output_file)
    write(structure_file, atoms, format="extxyz")

    # The manifest is a tool output, not something the caller reassembles by
    # hand: its order is load-bearing for build_interchange's atom-count check.
    manifest_file = os.path.join(working_dir, "components.json")
    with open(manifest_file, "w") as f:
        json.dump(
            {"components": [
                {"name": c["name"], "smiles": c["smiles"], "count": int(c["count"])}
                for c in components
            ]},
            f,
            indent=2,
        )

    total_mass_g = sum(
        _molar_mass(c["smiles"]) * int(c["count"]) for c in components
    ) / _AVOGADRO
    realized_density = total_mass_g / ((edge ** 3) * _CM3_PER_A3)

    return {
        "structure_file": structure_file,
        "manifest_file": manifest_file,
        "box": edge,
        "density": realized_density,
        "n_atoms": len(atoms),
        "n_molecules": sum(int(c["count"]) for c in components),
    }


TOOL_SPEC = ToolSpec(
    name="build_box",
    description=(
        "Pack a periodic box of component molecules (SMILES + counts) at a "
        "target density with Packmol, and write engine-neutral extxyz "
        "coordinates plus the components.json manifest build_interchange "
        "consumes. Reproducible: the conformer and packing RNGs are both "
        "seeded, so re-running reproduces the box and a series of calls "
        "differing only in counts yields boxes differing only in composition."
    ),
    parameters={
        "components": {
            "type": "list",
            "description": "[{name, smiles, count}] — packed and written to the manifest in this order",
        },
        "density": {
            "type": "number",
            "description": "target density g/cm^3 (e.g. 1.0 aqueous); required unless box is given",
        },
        "box": {
            "type": "number",
            "description": "cubic box edge in A; overrides density",
        },
        "working_dir": {
            "type": "string",
            "description": "where the coordinates, manifest, and Packmol scratch files are written",
        },
        "seed": {
            "type": "integer",
            "description": (
                "Packmol placement seed (default 1). Reproducibility knob: hold "
                "FIXED across a composition series so members differ only in "
                "composition; CHANGE it to draw an independent packing of the "
                "same composition (e.g. to estimate run-to-run spread)."
            ),
        },
        "conformer_seed": {
            "type": "integer",
            "description": "RDKit 3D-embedding seed (default 42); same rule as seed",
        },
        "tolerance": {
            "type": "number",
            "description": (
                "Packmol minimum inter-atomic separation in A (default 2.0). "
                "RAISE toward ~2.5 if the pack leaves close contacts the "
                "minimizer cannot relax; LOWER toward ~1.5 only for dense "
                "systems Packmol otherwise fails to fit."
            ),
        },
        "timeout": {
            "type": "number",
            "description": "seconds allowed for Packmol (default 300); RAISE for large or dense boxes",
        },
    },
    required=["components"],
    signature=(
        "build_box(components, density=None, box=None, working_dir='.', "
        "output_file='structure.extxyz', seed=1, conformer_seed=42, "
        "tolerance=2.0, timeout=300) -> dict"
    ),
    import_line=(
        "from scilink.skills.structure_generation.condensed.build_box import build_box"
    ),
    agents=["simulation"],
    when_to_use=(
        "When building a liquid, solution, or multi-component condensed box "
        "from known species and counts — especially for a composition or "
        "concentration series, where every member must be packed identically "
        "except for composition. Hand the returned structure_file and "
        "manifest_file straight to build_interchange."
    ),
    returns=(
        "dict with structure_file, manifest_file, box (A), density (realized "
        "g/cm^3), n_atoms, n_molecules"
    ),
    example=(
        "from scilink.skills.structure_generation.condensed.build_box import build_box\n\n"
        "# 40% cosolvent member of a series: only `count` changes between members.\n"
        "out = build_box(\n"
        "    components=[\n"
        "        {'name': 'water', 'smiles': 'O', 'count': 600},\n"
        "        {'name': 'cosolvent', 'smiles': 'CCS(=O)(=O)C(C)C', 'count': 80},\n"
        "    ],\n"
        "    density=1.05, working_dir='./member_40', seed=1,\n"
        ")"
    ),
)
