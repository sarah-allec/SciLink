"""PNNL-relevant benchmark systems.

One ``BenchmarkSystem`` per test target.  The same registry is read by
every runner — ``systems_for_agent("dft")`` etc. give each runner the
subset it cares about.

Structure types:
  * ``bulk``     — ASE ``Atoms`` from ``ase.build.bulk(...)`` or similar.
  * ``surface``  — slab built from a primitive + vacuum.
  * ``defected`` — bulk with one or more point defects.
  * ``liquid``   — Packmol build spec (no ASE Atoms upfront).
  * ``solute_in_solvent`` — biomolecule / ion in a Packmol-built solvent box.

The DFT-side ``expected`` block uses experimental room-temperature
conventional-cubic-cell values where they exist (Kittel etc.).  Surface
and adsorbate "expected" values are PBE references from the literature,
not experimental — we'll cite the source per system.

Goal levels:
  * ``guided`` — names the material class and nudges on the physics knobs
    the agent should think about (spin, +U, smearing).
  * ``bare``   — states only the system and the task; the agent must
    classify the material and infer every parameter itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Tier constants — every system belongs to exactly one.
TIER_BASELINE       = "baseline"          # carry-over from validation/
TIER_ENERGY_STORAGE = "energy_storage"
TIER_CATALYSIS      = "catalysis"
TIER_ENVIRONMENTAL  = "environmental"
TIER_AQUEOUS        = "aqueous"

GOAL_LEVELS = ("guided", "bare")

# Agent tags — drives systems_for_agent().
AGENTS_ALL = {"router", "structure_gen", "quality", "dft", "updater",
              "md", "mlip", "orchestrator"}


@dataclass(frozen=True)
class BenchmarkSystem:
    name: str                          # slug, used for output dirs
    tier: str                          # one of TIER_*
    kind: str                          # bulk | surface | defected | liquid | solute_in_solvent
    description: str                   # human-readable, one line
    elements: List[str]
    expected: Dict[str, Any]           # reference observables (a, gap, E_form, ρ, …)
    research_goals: Dict[str, str]     # GOAL_LEVELS → goal text
    tags: List[str]                    # ["magnetic", "+U", "spin-orbit", "vdW", …]
    agents: List[str]                  # subset of AGENTS_ALL — which runners test this
    builder: Optional[Callable[[], Any]] = None
    reference_source: str = ""         # citation for the expected values

    def goal(self, level: str = "guided") -> str:
        if level not in self.research_goals:
            raise KeyError(
                f"{self.name!r} has no {level!r} goal; "
                f"levels: {sorted(self.research_goals)}"
            )
        return self.research_goals[level]

    def build(self) -> Any:
        if self.builder is None:
            raise NotImplementedError(
                f"{self.name!r} has no builder yet — see TODOs in systems.py")
        return self.builder()


# ──────────────────────────────────────────────────────────────────
#  Builders — kept lightweight; expensive imports are local.
# ──────────────────────────────────────────────────────────────────

def _bulk_fe() -> "Atoms":
    from ase.build import bulk
    return bulk("Fe", "bcc", a=2.8665, cubic=True)


def _bulk_cu() -> "Atoms":
    from ase.build import bulk
    return bulk("Cu", "fcc", a=3.615, cubic=True)


def _bulk_si() -> "Atoms":
    from ase.build import bulk
    return bulk("Si", "diamond", a=5.431, cubic=True)


def _bulk_c_diamond() -> "Atoms":
    from ase.build import bulk
    return bulk("C", "diamond", a=3.567, cubic=True)


def _bulk_mgo() -> "Atoms":
    from ase.build import bulk
    return bulk("MgO", "rocksalt", a=4.212, cubic=True)


def _licoo2_layered() -> "Atoms":
    # Layered R-3m LiCoO2.  Use pymatgen if available for the proper
    # primitive; otherwise fall back to a hand-rolled hexagonal cell.
    # TODO: drop in the proper pymatgen build once we wire the import.
    raise NotImplementedError(
        "TODO: pymatgen MaterialsProject builder for LiCoO2 (mp-22526)")


def _pt111_co_top() -> "Atoms":
    from ase.build import fcc111, add_adsorbate
    slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0, a=3.924)
    add_adsorbate(slab, "C", height=1.85, position="ontop")
    add_adsorbate(slab, "O", height=3.00, position="ontop")
    slab.center(vacuum=12.0, axis=2)
    return slab


def _tio2_101_h2o() -> "Atoms":
    # Anatase TiO2 (101) slab with one H2O adsorbed near a 5-coord Ti.
    # TODO: pymatgen MP fetch + surface cut.
    raise NotImplementedError(
        "TODO: anatase TiO2(101) slab + H2O adsorbate")


def _uo2_bulk() -> "Atoms":
    # UO2, fluorite structure, a≈5.470 Å.  Hand-rolled — ase.build.bulk
    # doesn't ship a fluorite shortcut.
    from ase import Atoms
    a = 5.470
    return Atoms(
        symbols=["U", "U", "U", "U", "O", "O", "O", "O", "O", "O", "O", "O"],
        scaled_positions=[
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
            (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
            (0.75, 0.75, 0.75), (0.25, 0.25, 0.75),
            (0.25, 0.75, 0.25), (0.75, 0.25, 0.25),
        ],
        cell=[a, a, a],
        pbc=True,
    )


def _water_box_spec() -> dict:
    """Packmol build spec for ~900 H2O in a 30 Å cube (≈ 1 g/cm³)."""
    return {
        "kind": "packmol",
        "box": [30.0, 30.0, 30.0],
        "molecules": [{"name": "water", "count": 900, "smiles": "O"}],
        "target_density_g_cm3": 1.00,
    }


def _nacl_aq_spec() -> dict:
    # 1 M NaCl in water — ~ 18 ion pairs in a 1000-water box (30 Å cube).
    return {
        "kind": "packmol",
        "box": [30.0, 30.0, 30.0],
        "molecules": [
            {"name": "water", "count": 900, "smiles": "O"},
            {"name": "Na",    "count": 18,  "ion": "Na+"},
            {"name": "Cl",    "count": 18,  "ion": "Cl-"},
        ],
        "target_density_g_cm3": 1.02,
    }


def _lipf6_ec_spec() -> dict:
    # 1 M LiPF6 in ethylene carbonate.  EC density ≈ 1.32 g/cm³.
    return {
        "kind": "packmol",
        "box": [35.0, 35.0, 35.0],
        "molecules": [
            {"name": "ec",  "count": 280, "smiles": "C1COC(=O)O1"},
            {"name": "Li",  "count": 25,  "ion": "Li+"},
            {"name": "PF6", "count": 25,  "ion": "PF6-"},
        ],
        "target_density_g_cm3": 1.35,
    }


def _glycine_in_water_spec() -> dict:
    # Single zwitterionic glycine in a water box.
    return {
        "kind": "packmol",
        "box": [28.0, 28.0, 28.0],
        "molecules": [
            {"name": "glycine", "count": 1,
             "smiles": "C(C(=O)[O-])[NH3+]"},
            {"name": "water",   "count": 700, "smiles": "O"},
        ],
        "target_density_g_cm3": 1.00,
    }


# ──────────────────────────────────────────────────────────────────
#  Registry — single source of truth.  Order is intentional:
#  baseline tier first (continuity with validation/), then PNNL tiers.
# ──────────────────────────────────────────────────────────────────

REGISTRY: List[BenchmarkSystem] = [

    # ── Baseline (carry-over from validation/) ────────────────────
    BenchmarkSystem(
        name="fe_bcc",
        tier=TIER_BASELINE,
        kind="bulk",
        description="α-iron, BCC (Im-3m), ferromagnetic",
        elements=["Fe"],
        expected={"a": 2.8665, "magnetic_moment_per_atom": 2.22},
        research_goals={
            "guided": "Relax cell and ions of BCC iron.  This is a "
                      "spin-polarized metal — make sure the calculation "
                      "is spin-polarized and the smearing / k-point density "
                      "are appropriate for a magnetic metal.  Report the "
                      "equilibrium lattice constant.",
            "bare":   "Relax the cell and ions of BCC iron and report "
                      "the equilibrium lattice constant.",
        },
        tags=["magnetic", "metal"],
        agents=["dft", "mlip", "structure_gen"],
        builder=_bulk_fe,
        reference_source="Kittel, ISSP 8th ed.",
    ),
    BenchmarkSystem(
        name="cu_fcc",
        tier=TIER_BASELINE,
        kind="bulk",
        description="copper, FCC (Fm-3m)",
        elements=["Cu"],
        expected={"a": 3.615},
        research_goals={
            "guided": "Relax the cell and ions of FCC copper.  Use "
                      "smearing appropriate for a metal and a dense enough "
                      "k-point grid.  Report the equilibrium lattice "
                      "constant.",
            "bare":   "Relax the cell and ions of FCC copper and report "
                      "the equilibrium lattice constant.",
        },
        tags=["metal"],
        agents=["dft", "mlip", "structure_gen"],
        builder=_bulk_cu,
        reference_source="Kittel, ISSP 8th ed.",
    ),
    BenchmarkSystem(
        name="si_diamond",
        tier=TIER_BASELINE,
        kind="bulk",
        description="silicon, diamond cubic (Fd-3m)",
        elements=["Si"],
        expected={"a": 5.431, "gap_indirect": 1.17},
        research_goals={
            "guided": "Relax cell and ions of diamond-cubic silicon.  "
                      "This is a covalent semiconductor — use Gaussian "
                      "smearing with small sigma and a converged k-point "
                      "grid.  Report the equilibrium lattice constant.",
            "bare":   "Relax cell and ions of silicon in the diamond "
                      "structure and report the equilibrium lattice "
                      "constant.",
        },
        tags=["semiconductor", "covalent"],
        agents=["dft", "mlip", "structure_gen"],
        builder=_bulk_si,
        reference_source="Kittel, ISSP 8th ed.",
    ),
    BenchmarkSystem(
        name="c_diamond",
        tier=TIER_BASELINE,
        kind="bulk",
        description="diamond carbon (Fd-3m)",
        elements=["C"],
        expected={"a": 3.567, "gap_indirect": 5.47},
        research_goals={
            "guided": "Relax cell and ions of diamond carbon.  Wide-gap "
                      "insulator — choose smearing and ENCUT appropriately. "
                      "Report the equilibrium lattice constant.",
            "bare":   "Relax cell and ions of diamond carbon and report "
                      "the equilibrium lattice constant.",
        },
        tags=["insulator", "covalent"],
        agents=["dft", "structure_gen"],
        builder=_bulk_c_diamond,
        reference_source="Kittel, ISSP 8th ed.",
    ),
    BenchmarkSystem(
        name="mgo_rocksalt",
        tier=TIER_BASELINE,
        kind="bulk",
        description="magnesium oxide, rock-salt (Fm-3m)",
        elements=["Mg", "O"],
        expected={"a": 4.212},
        research_goals={
            "guided": "Relax cell and ions of rock-salt MgO.  Ionic "
                      "insulator — pick smearing accordingly.  Report "
                      "the equilibrium lattice constant.",
            "bare":   "Relax cell and ions of rock-salt MgO and report "
                      "the equilibrium lattice constant.",
        },
        tags=["insulator", "ionic"],
        agents=["dft", "mlip", "structure_gen"],
        builder=_bulk_mgo,
        reference_source="Kittel, ISSP 8th ed.",
    ),

    # ── Energy storage ────────────────────────────────────────────
    BenchmarkSystem(
        name="licoo2_layered",
        tier=TIER_ENERGY_STORAGE,
        kind="bulk",
        description="lithium cobalt oxide, layered R-3m (battery cathode)",
        elements=["Li", "Co", "O"],
        expected={"a": 2.815, "c": 14.05},   # hexagonal lattice
        research_goals={
            "guided": "Relax the layered R-3m LiCoO2 unit cell.  Co is "
                      "in a low-spin d6 state in this structure; standard "
                      "PBE underestimates the band gap — apply a Hubbard "
                      "U on Co (Dudarev formulation, U ~ 4 eV is typical). "
                      "Report the relaxed a and c lattice parameters.",
            "bare":   "Relax the LiCoO2 cathode crystal and report the "
                      "lattice parameters.",
        },
        tags=["magnetic", "+U", "layered", "battery"],
        agents=["dft", "structure_gen", "orchestrator", "mlip"],
        builder=_licoo2_layered,
        reference_source="MP mp-22526; Aydinol et al., PRB 56, 1354 (1997).",
    ),
    BenchmarkSystem(
        name="lipf6_in_ec",
        tier=TIER_ENERGY_STORAGE,
        kind="liquid",
        description="1 M LiPF6 in ethylene carbonate (Li-ion battery electrolyte)",
        elements=["Li", "P", "F", "C", "H", "O"],
        expected={"density_g_cm3": 1.35},     # ≈ 1.32–1.38 for pure EC + 1 M salt
        research_goals={
            "guided": "Build a periodic box of LiPF6 in ethylene carbonate "
                      "at ~1 M (≈ 25 ion pairs in ~280 EC molecules).  "
                      "Equilibrate at 300 K and 1 atm; report density and "
                      "Li-O(carbonyl) radial distribution.",
            "bare":   "Set up an MD simulation of 1 M LiPF6 in ethylene "
                      "carbonate at room temperature.",
        },
        tags=["md", "electrolyte", "packmol"],
        agents=["md", "orchestrator"],
        builder=_lipf6_ec_spec,
        reference_source="Borodin & Smith, J. Phys. Chem. B 113, 1763 (2009).",
    ),

    # ── Catalysis ─────────────────────────────────────────────────
    BenchmarkSystem(
        name="pt111_co_top",
        tier=TIER_CATALYSIS,
        kind="surface",
        description="Pt(111) 3×3 slab, 4 layers, CO adsorbed at top site",
        elements=["Pt", "C", "O"],
        expected={
            "E_ads_eV": -1.45,   # PBE; experiment is ~ -1.7 eV
            "Pt_Pt_a_bulk": 3.924,
        },
        research_goals={
            "guided": "Relax a CO molecule adsorbed at the top site of "
                      "a Pt(111) 3×3 / 4-layer slab.  Metallic surface — "
                      "use Methfessel-Paxton smearing.  Freeze the bottom "
                      "two layers; relax the rest.  Report the adsorption "
                      "energy and the Pt-C, C-O bond lengths.",
            "bare":   "Compute the CO adsorption energy at the top site "
                      "of Pt(111).",
        },
        tags=["surface", "adsorption", "metal"],
        agents=["dft", "structure_gen", "orchestrator"],
        builder=_pt111_co_top,
        reference_source="Feibelman et al., J. Phys. Chem. B 105, 4018 (2001).",
    ),
    BenchmarkSystem(
        name="tio2_101_h2o",
        tier=TIER_CATALYSIS,
        kind="surface",
        description="anatase TiO2(101) with one H2O adsorbed at a 5-coord Ti",
        elements=["Ti", "O", "H"],
        expected={"E_ads_eV": -0.74},   # PBE molecular adsorption
        research_goals={
            "guided": "Adsorb a single H2O molecule on the anatase "
                      "TiO2(101) surface near a 5-coordinate Ti site.  "
                      "Use the standard 3×1 slab terminated by stoichiometric "
                      "TiO2.  Report molecular vs dissociative adsorption "
                      "energies if both are stable.",
            "bare":   "Compute the adsorption energy of water on the "
                      "(101) surface of anatase TiO2.",
        },
        tags=["surface", "adsorption", "oxide"],
        agents=["dft", "structure_gen"],
        builder=_tio2_101_h2o,
        reference_source="Vittadini et al., PRL 81, 2954 (1998).",
    ),

    # ── Environmental / actinide ──────────────────────────────────
    BenchmarkSystem(
        name="uo2_fluorite",
        tier=TIER_ENVIRONMENTAL,
        kind="bulk",
        description="uranium dioxide, fluorite (Fm-3m), antiferromagnetic ground state",
        elements=["U", "O"],
        expected={"a": 5.470, "U_moment_uB": 1.7},
        research_goals={
            "guided": "Relax UO2 in the fluorite structure.  This is a "
                      "Mott insulator with f-electrons — apply Hubbard U "
                      "on U-5f (Dudarev, U ≈ 4 eV) and use 1k AFM ordering "
                      "along [001].  Report the relaxed lattice constant.",
            "bare":   "Relax uranium dioxide in its standard crystal "
                      "structure and report the lattice constant.",
        },
        tags=["magnetic", "+U", "f-electron", "actinide"],
        agents=["dft", "structure_gen"],
        builder=_uo2_bulk,
        reference_source="Dorado et al., PRB 79, 235125 (2009).",
    ),

    # ── Aqueous + biomolecular ────────────────────────────────────
    BenchmarkSystem(
        name="water_box",
        tier=TIER_AQUEOUS,
        kind="liquid",
        description="liquid water at ambient conditions",
        elements=["H", "O"],
        expected={"density_g_cm3": 1.00,
                  "RDF_OO_first_peak_A": 2.77,
                  "self_diffusion_1e-9_m2_s": 2.3},
        research_goals={
            "guided": "Build a periodic box of ~900 water molecules at "
                      "~1 g/cm³.  Equilibrate at 300 K and 1 atm with a "
                      "rigid water model (TIP3P/SPC).  Report density "
                      "and the O-O RDF first-peak position.",
            "bare":   "Run an MD simulation of liquid water at 300 K.",
        },
        tags=["md", "mlip", "polar-liquid"],
        agents=["md", "mlip", "orchestrator"],
        builder=_water_box_spec,
        reference_source="Soper, Chem. Phys. 258, 121 (2000).",
    ),
    BenchmarkSystem(
        name="nacl_aq_1M",
        tier=TIER_AQUEOUS,
        kind="liquid",
        description="1 M NaCl in water",
        elements=["Na", "Cl", "H", "O"],
        expected={"density_g_cm3": 1.038,
                  "RDF_NaO_first_peak_A": 2.40,
                  "RDF_ClO_first_peak_A": 3.20},
        research_goals={
            "guided": "Build a 1 M aqueous NaCl box (~ 18 ion pairs in "
                      "900 water).  Equilibrate at 300 K and 1 atm.  "
                      "Report density and Na-O / Cl-O first-shell radii.",
            "bare":   "Run an MD simulation of 1 M aqueous NaCl at "
                      "room temperature.",
        },
        tags=["md", "electrolyte", "ions"],
        agents=["md"],
        builder=_nacl_aq_spec,
        reference_source="Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008).",
    ),
    BenchmarkSystem(
        name="glycine_in_water",
        tier=TIER_AQUEOUS,
        kind="solute_in_solvent",
        description="one zwitterionic glycine in a water box",
        elements=["C", "H", "N", "O"],
        expected={"density_g_cm3": 1.00,
                  "glycine_radius_of_gyration_A": 1.8},
        research_goals={
            "guided": "Solvate a single zwitterionic glycine (NH3+-CH2-COO-) "
                      "in a ~700-water box at 1 g/cm³.  Equilibrate at "
                      "300 K and 1 atm.  Report glycine Rg and the "
                      "RDF between glycine N/O and water O.",
            "bare":   "Run an MD simulation of glycine in water at "
                      "room temperature.",
        },
        tags=["md", "biomolecular", "zwitterion"],
        agents=["md", "orchestrator"],
        builder=_glycine_in_water_spec,
        reference_source="Hamad et al., J. Phys. Chem. B 109, 15499 (2005).",
    ),
]


# ──────────────────────────────────────────────────────────────────
#  Accessors
# ──────────────────────────────────────────────────────────────────

def get_system(name: str) -> BenchmarkSystem:
    for s in REGISTRY:
        if s.name == name:
            return s
    raise KeyError(
        f"unknown system {name!r};  known: {[s.name for s in REGISTRY]}")


def systems_in_tier(tier: str) -> List[BenchmarkSystem]:
    return [s for s in REGISTRY if s.tier == tier]


def systems_for_agent(agent: str) -> List[BenchmarkSystem]:
    if agent not in AGENTS_ALL:
        raise KeyError(
            f"unknown agent tag {agent!r};  known: {sorted(AGENTS_ALL)}")
    return [s for s in REGISTRY if agent in s.agents]


def all_systems() -> List[BenchmarkSystem]:
    return list(REGISTRY)


if __name__ == "__main__":
    # quick sanity print: roster overview
    print(f"{'name':<22} {'tier':<18} {'kind':<20} agents")
    print("-" * 88)
    for s in REGISTRY:
        print(f"{s.name:<22} {s.tier:<18} {s.kind:<20} {','.join(s.agents)}")
