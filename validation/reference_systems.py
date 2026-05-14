"""
Curated panel of bulk crystals with well-documented experimental
lattice constants — the reference set for the "agents produce correct
physics" validation.

Each system is handed to the agent as a natural-language goal; the
agent generates relaxation inputs; the run relaxes the cell; we
compare the relaxed lattice constant to the experimental value.

The structures are intentionally built at 0.97× the experimental
lattice constant (``build(scale=0.97)``) so the relaxation has to do
real work — expand a ~9%-compressed cell back to equilibrium. This
tests that the agent's *generated inputs actually drive a correct
relaxation*, not merely that a pre-optimized structure stayed put.

Expected outcomes:
  - VASP/PBE: typically overestimates lattice constants by ~0.5-1.5%
    (well-known PBE behavior). Landing in that band IS the correct
    physics — the agent isn't being asked to beat PBE, only to
    drive PBE correctly.
  - MACE-mp-0 / CHGNet: both trained on PBE (MPtrj) data, so they
    should reproduce the PBE-equilibrium lattice constant, hence
    also ~0.5-1.5% above experiment.

So the validation table's "correct" band is roughly
[experiment, experiment + 2%] for all three engines, and all three
should agree with each other to within a few tenths of a percent.
"""

from dataclasses import dataclass
from typing import Callable, List

from ase import Atoms
from ase.build import bulk


@dataclass(frozen=True)
class ReferenceSystem:
    """One bulk crystal + its experimental reference + the agent goal."""

    name: str                          # slug, used for output dirs
    crystal: str                       # human-readable structure label
    elements: List[str]
    exp_lattice_constant: float        # Angstrom, conventional cubic cell
    reference_source: str              # citation for the reference value
    research_goal: str                 # natural-language goal given to the agent
    _builder: Callable[[], Atoms]      # returns the conventional cubic cell

    def build(self, scale: float = 0.97) -> Atoms:
        """Return the structure scaled by ``scale`` relative to experiment.

        ``scale < 1`` compresses the cell so the relaxation has to
        expand it back — the actual test of the generated inputs.
        ``scale = 1.0`` gives the structure exactly at the
        experimental lattice constant (useful for a sanity pass).
        """
        atoms = self._builder()
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        return atoms

    @property
    def n_atoms_conventional(self) -> int:
        return len(self._builder())


def _si() -> Atoms:
    return bulk("Si", "diamond", a=5.431, cubic=True)


def _cu() -> Atoms:
    return bulk("Cu", "fcc", a=3.615, cubic=True)


def _mgo() -> Atoms:
    return bulk("MgO", "rocksalt", a=4.212, cubic=True)


def _diamond() -> Atoms:
    return bulk("C", "diamond", a=3.567, cubic=True)


# The panel. Experimental lattice constants are room-temperature
# conventional-cubic-cell values from Kittel's standard reference;
# they are the textbook numbers every solid-state course uses, which
# is exactly why they make an unambiguous "correct physics" target.
PANEL: List[ReferenceSystem] = [
    ReferenceSystem(
        name="si_diamond",
        crystal="silicon, diamond cubic (Fd-3m)",
        elements=["Si"],
        exp_lattice_constant=5.431,
        reference_source=(
            "Kittel, Introduction to Solid State Physics, 8th ed., "
            "Table 1 (room-temperature a)."
        ),
        research_goal=(
            "Relax bulk silicon in the diamond-cubic structure to its "
            "equilibrium geometry (full cell + ionic relaxation) and "
            "report the optimized lattice constant. This is a "
            "semiconductor; choose smearing and k-point density "
            "accordingly."
        ),
        _builder=_si,
    ),
    ReferenceSystem(
        name="cu_fcc",
        crystal="copper, face-centered cubic (Fm-3m)",
        elements=["Cu"],
        exp_lattice_constant=3.615,
        reference_source=(
            "Kittel, Introduction to Solid State Physics, 8th ed., "
            "Table 1 (room-temperature a)."
        ),
        research_goal=(
            "Relax bulk FCC copper to its equilibrium geometry (full "
            "cell + ionic relaxation) and report the optimized lattice "
            "constant. This is a metal; use appropriate smearing and a "
            "dense k-point mesh."
        ),
        _builder=_cu,
    ),
    ReferenceSystem(
        name="mgo_rocksalt",
        crystal="magnesium oxide, rock-salt (Fm-3m)",
        elements=["Mg", "O"],
        exp_lattice_constant=4.212,
        reference_source=(
            "Kittel, Introduction to Solid State Physics, 8th ed., "
            "Table 1 (room-temperature a)."
        ),
        research_goal=(
            "Relax bulk magnesium oxide in the rock-salt structure to "
            "its equilibrium geometry (full cell + ionic relaxation) "
            "and report the optimized lattice constant. This is a wide-"
            "gap ionic insulator."
        ),
        _builder=_mgo,
    ),
    ReferenceSystem(
        name="c_diamond",
        crystal="diamond carbon (Fd-3m)",
        elements=["C"],
        exp_lattice_constant=3.567,
        reference_source=(
            "Kittel, Introduction to Solid State Physics, 8th ed., "
            "Table 1 (room-temperature a)."
        ),
        research_goal=(
            "Relax bulk diamond (cubic carbon) to its equilibrium "
            "geometry (full cell + ionic relaxation) and report the "
            "optimized lattice constant. Carbon is a light element — "
            "ensure the plane-wave cutoff is high enough."
        ),
        _builder=_diamond,
    ),
]


# Engines the validation harness will route each system through. VASP
# is periodic DFT; MACE and CHGNet are MLIP backends (ASE runner).
# QE is added once its skill bundle lands (Monday P2).
ENGINES = {
    "vasp":   {"scale": "periodic_dft", "agent": "PeriodicDFTAgent"},
    "mace":   {"scale": "machine_learning_potentials", "agent": "MLIPAgent"},
    "chgnet": {"scale": "machine_learning_potentials", "agent": "MLIPAgent"},
}


def get_system(name: str) -> ReferenceSystem:
    """Look up a ReferenceSystem by its slug name."""
    for sys in PANEL:
        if sys.name == name:
            return sys
    raise KeyError(
        f"unknown system {name!r}; panel has: "
        f"{[s.name for s in PANEL]}"
    )


if __name__ == "__main__":
    # Quick sanity dump
    print(f"{'system':<16} {'crystal':<38} {'exp a (Å)':>10} {'atoms':>6}")
    print("-" * 74)
    for sys in PANEL:
        print(
            f"{sys.name:<16} {sys.crystal:<38} "
            f"{sys.exp_lattice_constant:>10.3f} "
            f"{sys.n_atoms_conventional:>6d}"
        )
    print()
    print(f"engines: {list(ENGINES)}")
    print(f"compressed-cell sanity (Si @ 0.97x):")
    si = get_system("si_diamond")
    a0 = si.build(scale=1.0).cell.lengths()[0]
    a_compressed = si.build(scale=0.97).cell.lengths()[0]
    print(f"  experimental a = {a0:.4f} Å")
    print(f"  compressed   a = {a_compressed:.4f} Å  "
          f"({100*(a_compressed/a0 - 1):.1f}%)")
