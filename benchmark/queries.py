"""Router test prompts — natural-language goals + ground-truth labels.

Each entry is a ``RouterQuery`` with:
  * ``prompt``           — the user's natural-language ask, as a SciLink
                           user might type it.
  * ``expected_scale``   — the scale the router should land on.
                           One of ``periodic_dft`` / ``molecular_dft`` /
                           ``molecular_dynamics`` / ``machine_learning_potentials``.
  * ``acceptable_engines`` — any engine in this list is scored correct
                           at that scale (some prompts are engine-agnostic).
  * ``difficulty``       — ``easy`` / ``medium`` / ``hard``.  Drives a
                           per-difficulty accuracy breakdown.
  * ``notes``            — what makes the prompt interesting.

Difficulty rubric
-----------------
  easy    : the scale and (often) engine are stated outright or strongly
            implied.  A correct router should land it without ambiguity.
  medium  : the scale follows from the system + observable, but the agent
            has to combine those signals.
  hard    : the prompt is genuinely ambiguous (multiple valid routes)
            or under-specified.  We score the agent against the most
            common practitioner choice and accept alternatives via
            ``acceptable_engines``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RouterQuery:
    id: str
    prompt: str
    expected_scale: str
    acceptable_engines: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""
    # Capability-gap signal — set when the expected scale needs an agent
    # that isn't yet in the codebase (e.g. "molecular_dft" — no
    # MolecularDFTAgent exists today).  Such queries are reported in a
    # separate "capability gap" tally rather than counted against the
    # router's accuracy.
    requires_agent: Optional[str] = None
    # For genuinely ambiguous prompts where multiple scales are valid
    # modern routes (e.g. DFT phonons vs MLIP-MD for thermal
    # conductivity), list every scale we'd accept.  Defaults to
    # [expected_scale] when empty.
    acceptable_scales: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
#  Periodic DFT — bulks, surfaces, defects.
# ──────────────────────────────────────────────────────────────────
PERIODIC_DFT: List[RouterQuery] = [
    RouterQuery(
        id="pd_01_lattice_cu",
        prompt="What is the equilibrium lattice constant of FCC copper?",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="easy",
        notes="textbook bulk DFT; any planewave engine acceptable",
    ),
    RouterQuery(
        id="pd_02_band_gap_si",
        prompt="Calculate the indirect band gap of silicon in the "
               "diamond structure.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="easy",
        notes="bulk semiconductor property",
    ),
    RouterQuery(
        id="pd_03_vacancy_mgo",
        prompt="Compute the formation energy of an oxygen vacancy in MgO.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="medium",
        notes="periodic defect calc",
    ),
    RouterQuery(
        id="pd_04_surface_pt_co",
        prompt="What is the CO adsorption energy on the Pt(111) surface "
               "at the top site?",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="medium",
        notes="slab DFT + adsorbate — classic catalysis question",
    ),
    RouterQuery(
        id="pd_05_perovskite_relax",
        prompt="Relax the BaTiO3 cubic perovskite unit cell and "
               "report the equilibrium lattice parameter.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="easy",
        notes="multi-element bulk; agent should pick periodic_dft naturally",
    ),
    RouterQuery(
        id="pd_06_actinide_uo2",
        prompt="Find the equilibrium lattice constant of UO2 in the "
               "fluorite structure with antiferromagnetic ordering.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="medium",
        notes="f-electron + AFM hint; bulk DFT",
    ),
    RouterQuery(
        id="pd_07_battery_licoo2",
        prompt="Relax the LiCoO2 cathode crystal in its layered R-3m phase.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="easy",
        notes="battery material; multi-element bulk",
    ),

    # ── Periodic DFT, harder / ambiguous ──────────────────────────
    RouterQuery(
        id="pd_08_screen_oxide",
        prompt="I want to predict whether a new ternary oxide is stable "
               "with respect to its competing phases.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="hard",
        notes="formation-energy screen; could be DFT or MLIP, but "
              "'predict stability' is conventionally DFT",
    ),
]


# ──────────────────────────────────────────────────────────────────
#  Molecular DFT — gas-phase / cluster.
# ──────────────────────────────────────────────────────────────────
MOLECULAR_DFT: List[RouterQuery] = [
    RouterQuery(
        id="md_01_dipole_formaldehyde",
        prompt="Compute the dipole moment of formaldehyde in the gas phase.",
        expected_scale="molecular_dft",
        acceptable_engines=["pyscf", "orca", "gaussian", "nwchem"],
        difficulty="easy",
        notes="gas-phase molecular property — Gaussian-basis territory",
        requires_agent="molecular_dft",
    ),
    RouterQuery(
        id="md_02_homo_lumo_benzene",
        prompt="Calculate the HOMO-LUMO gap of benzene at the "
               "B3LYP/6-31G(d) level.",
        expected_scale="molecular_dft",
        acceptable_engines=["pyscf", "orca", "gaussian", "nwchem"],
        difficulty="easy",
        notes="basis set named explicitly → molecular_dft",
        requires_agent="molecular_dft",
    ),
    RouterQuery(
        id="md_03_co_binding",
        prompt="Calculate the binding energy of a CO dimer at the "
               "CCSD(T)/aug-cc-pVTZ level.",
        expected_scale="molecular_dft",
        acceptable_engines=["pyscf", "orca", "gaussian", "nwchem"],
        difficulty="easy",
        notes="post-HF method → cluster code",
        requires_agent="molecular_dft",
    ),
]


# ──────────────────────────────────────────────────────────────────
#  Classical MD — liquids, biomolecules, longer timescales.
# ──────────────────────────────────────────────────────────────────
CLASSICAL_MD: List[RouterQuery] = [
    RouterQuery(
        id="cmd_01_water_box",
        prompt="Run a molecular dynamics simulation of liquid water at "
               "300 K and 1 atm.",
        expected_scale="molecular_dynamics",
        acceptable_engines=["lammps", "gromacs", "openmm", "amber"],
        difficulty="easy",
        notes="classic MD prompt; any engine fine",
    ),
    RouterQuery(
        id="cmd_02_nacl_electrolyte",
        prompt="Set up an MD simulation of 1 M NaCl in water at "
               "room temperature.",
        expected_scale="molecular_dynamics",
        acceptable_engines=["lammps", "gromacs", "openmm", "amber"],
        difficulty="easy",
        notes="electrolyte MD",
    ),
    RouterQuery(
        id="cmd_03_battery_electrolyte",
        prompt="Build an MD simulation of 1 M LiPF6 in ethylene carbonate.",
        expected_scale="molecular_dynamics",
        acceptable_engines=["lammps", "gromacs", "openmm", "amber"],
        difficulty="medium",
        notes="battery electrolyte; multi-component liquid",
    ),
    RouterQuery(
        id="cmd_04_peptide_unfolding",
        prompt="Simulate the unfolding of a 50-residue peptide in "
               "explicit water for 100 ns.",
        expected_scale="molecular_dynamics",
        acceptable_engines=["gromacs", "openmm", "amber", "lammps"],
        difficulty="medium",
        notes="biomolecular MD — long-timescale, force-field territory",
    ),
    RouterQuery(
        id="cmd_05_glycine_solvated",
        prompt="Run MD of a zwitterionic glycine in a water box at 300 K.",
        expected_scale="molecular_dynamics",
        acceptable_engines=["lammps", "gromacs", "openmm", "amber"],
        difficulty="easy",
        notes="solute in solvent; classic FF + Packmol",
    ),
]


# ──────────────────────────────────────────────────────────────────
#  MLIP-driven dynamics — foundation models / NN potentials.
# ──────────────────────────────────────────────────────────────────
MLIP: List[RouterQuery] = [
    # All three are deliberately backend-agnostic — they signal "use an
    # ML potential" but don't name MACE / CHGNet / NequIP.  Tests whether
    # the agent's backend selection varies by system (water vs battery
    # vs metal) or just defaults to one model regardless.  On the
    # validation panel the agent unanimously picked mace; this lets us
    # see whether richer system context shifts that.
    RouterQuery(
        id="mlip_01_water",
        prompt="Run molecular dynamics of liquid water using a pretrained "
               "machine-learning interatomic potential.",
        expected_scale="machine_learning_potentials",
        acceptable_engines=["mace", "chgnet", "nequip"],
        difficulty="medium",
        notes="ML potential signal but no backend named; agent picks",
    ),
    RouterQuery(
        id="mlip_02_battery_cathode",
        prompt="Run molecular dynamics on a LiCoO2 cathode cell at "
               "elevated temperature with a pretrained ML potential.",
        expected_scale="machine_learning_potentials",
        acceptable_engines=["mace", "chgnet", "nequip"],
        difficulty="medium",
        notes="battery-cathode hint; agent could prefer chgnet (trained "
              "on MPF) — does it?",
    ),
    RouterQuery(
        id="mlip_03_iron_bulk",
        prompt="Run molecular dynamics with a neural-network interatomic "
               "potential on a 1000-atom iron system.",
        expected_scale="machine_learning_potentials",
        acceptable_engines=["mace", "chgnet", "nequip"],
        difficulty="medium",
        notes="generic NNP language; magnetic metal context",
    ),
]


# ──────────────────────────────────────────────────────────────────
#  Cross-scale / genuinely ambiguous — these score harder and we
#  accept the practitioner-conventional pick.
# ──────────────────────────────────────────────────────────────────
AMBIGUOUS: List[RouterQuery] = [
    RouterQuery(
        id="amb_01_li_diffusion",
        prompt="Calculate the diffusion coefficient of Li in LiCoO2 at "
               "elevated temperature.",
        expected_scale="machine_learning_potentials",
        acceptable_engines=["mace", "nequip", "chgnet"],
        difficulty="hard",
        notes="could be NEB-DFT (single hop) or MLIP-MD (full diffusivity); "
              "we accept MLIP as the conventional modern pick for D(T)",
    ),
    RouterQuery(
        id="amb_02_thermal_conductivity_si",
        prompt="Compute the lattice thermal conductivity of silicon at 300 K.",
        expected_scale="machine_learning_potentials",
        acceptable_scales=["machine_learning_potentials", "periodic_dft",
                           "molecular_dynamics"],
        acceptable_engines=["mace", "nequip", "chgnet",
                            "vasp", "qe", "abinit", "cp2k",
                            "lammps", "gromacs", "openmm"],
        difficulty="hard",
        notes="DFT-phonon BTE *or* MLIP-MD Green-Kubo are both valid "
              "modern routes for κ(T); classical MD with an empirical "
              "potential is also defensible. All three accepted.",
    ),
    RouterQuery(
        id="amb_03_melting_point_cu",
        prompt="Predict the melting point of copper from a two-phase "
               "molecular dynamics simulation.",
        expected_scale="machine_learning_potentials",
        acceptable_engines=["mace", "nequip", "chgnet"],
        difficulty="hard",
        notes="needs both fcc+liquid stable — accurate potential required",
    ),
    RouterQuery(
        id="amb_04_catalyst_screen",
        prompt="Screen transition-metal alloys for the oxygen-reduction "
               "reaction on a (111) surface.",
        expected_scale="periodic_dft",
        acceptable_engines=["vasp", "qe", "abinit", "cp2k"],
        difficulty="hard",
        notes="adsorption-energy screen — periodic_dft is the standard",
    ),
    RouterQuery(
        id="amb_05_water_at_interface",
        prompt="Study the structure of liquid water in contact with a "
               "TiO2 surface.",
        expected_scale="molecular_dynamics",
        acceptable_scales=["molecular_dynamics", "periodic_dft",
                           "machine_learning_potentials"],
        acceptable_engines=["lammps", "gromacs", "openmm", "amber",
                            "vasp", "qe", "abinit", "cp2k",
                            "mace", "nequip", "chgnet"],
        difficulty="hard",
        notes="AIMD (DFT-MD), classical MD, or MLIP-MD are all "
              "published routes for liquid–solid interfaces — accept any",
    ),
]


# ──────────────────────────────────────────────────────────────────
#  Aggregator
# ──────────────────────────────────────────────────────────────────

ALL_QUERIES: List[RouterQuery] = (
    PERIODIC_DFT + MOLECULAR_DFT + CLASSICAL_MD + MLIP + AMBIGUOUS
)


def by_scale(scale: str) -> List[RouterQuery]:
    return [q for q in ALL_QUERIES if q.expected_scale == scale]


def by_difficulty(difficulty: str) -> List[RouterQuery]:
    return [q for q in ALL_QUERIES if q.difficulty == difficulty]


if __name__ == "__main__":
    print(f"total: {len(ALL_QUERIES)} queries")
    for tag, group in [
        ("periodic_dft",                PERIODIC_DFT),
        ("molecular_dft",               MOLECULAR_DFT),
        ("classical MD",                CLASSICAL_MD),
        ("MLIP",                        MLIP),
        ("ambiguous / cross-scale",     AMBIGUOUS),
    ]:
        easy = sum(q.difficulty == "easy" for q in group)
        med  = sum(q.difficulty == "medium" for q in group)
        hard = sum(q.difficulty == "hard" for q in group)
        print(f"  {tag:<25} n={len(group):>2}   easy={easy} medium={med} hard={hard}")
