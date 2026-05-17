---
description: VASP DFT input generation — INCAR parameter selection (functional, smearing, spin polarization, parallelization) and KPOINTS conventions for metals, semiconductors, slabs, molecules, and NEB calculations.
detect:
  binaries: [vasp_std, vasp_gam, vasp_ncl, vasp]
  env_vars: [VASP_HOME, VASP_DIR]
  python_modules: []
  guidance: |
    VASP ships several binaries flavored by k-point sampling:
    vasp_std (general k-points), vasp_gam (gamma-only, fastest for
    large cells), vasp_ncl (noncollinear / spin-orbit). Some sites
    install them under $VASP_HOME/<flavor> or $VASP_DIR/bin/. On
    HPC clusters with Lmod or similar, the binary may only be on
    $PATH after `module load vasp/<version>`. Detection should
    consider any of the above as a positive hit.
---
# VASP Input Generation Skill

## overview

Density Functional Theory (DFT) calculations using the Vienna Ab initio
Simulation Package (VASP). This skill covers INCAR parameter selection
and KPOINTS generation for metallic systems, surfaces, molecules,
and transition state calculations. The goal is to produce input files
that are physically correct, computationally efficient, and consistent
with standard practices in the computational materials science literature.

## planning

**Functional selection:** The GGA tag in the INCAR controls which
pseudopotential directory ASE uses. This is critical:
- GGA = PE means PBE functional, ASE looks in potpaw_PBE/
- GGA = 91 means PW91 functional, ASE looks in potpaw_GGA/
- No GGA tag means LDA, ASE looks in potpaw/
ALWAYS include the GGA tag explicitly. Omitting it when PBE potentials
are intended causes a fatal "No pseudopotential" error because ASE
searches the wrong directory.

**System identification:** Before choosing parameters, identify the system:
1. Is it a metal, semiconductor, insulator, or molecule?
2. Is it bulk, a surface slab, or an isolated molecule/cluster?
3. Does it contain magnetic elements (Ni, Fe, Co, Mn, Cr)?
4. Does it contain hydrogen (requires adequate ENCUT)?
5. Is it a relaxation, single-point, NEB, or MD calculation?

**Smearing:** The choice of ISMEAR is dictated primarily by the
system type, NOT by whether the calculation is static or a
relaxation. The system type rule comes first; the
"tetrahedron-for-accurate-energies" exception applies only to
insulators.

- Metals and metallic surfaces: ISMEAR = 1 or 2 (Methfessel-Paxton),
  SIGMA = 0.1-0.2 eV. This is the right choice for metals in BOTH
  static SCF and relaxation calculations. Check that the entropy
  term T*S is less than 1 meV/atom in the OUTCAR. Do NOT default to
  ISMEAR = -5 for metals just because the calculation is static --
  tetrahedron is brittle for metals (sensitive to k-mesh symmetry,
  problems at Gamma) and is only worth the cost when you are
  specifically computing tetrahedron-method DOS or band-structure
  data on a fully-converged k-mesh.
- Semiconductors and insulators (relaxations): ISMEAR = 0 (Gaussian),
  SIGMA = 0.05 eV.
- Semiconductors and insulators (static, NSW = 0): ISMEAR = -5
  (tetrahedron with Blochl corrections) is preferred for accurate
  total energies and DOS. ISMEAR = 0 is also acceptable.
- Molecules in a box: ISMEAR = 0 (Gaussian), SIGMA = 0.01-0.05 eV.

NEVER use ISMEAR = -5 for an ionic relaxation or MD run regardless
of system type -- it produces erratic forces and typically fails
to converge.

**Spin polarization:** Systems containing Ni, Fe, Co, Mn, or Cr MUST
use ISPIN = 2 with appropriate MAGMOM initial values. Common initial
MAGMOM values:
- Ni: 2.0 muB per atom
- Fe: 5.0 muB per atom
- Co: 3.0 muB per atom
- Mn: 5.0 muB per atom
- Cr: 3.0 muB per atom
- Non-magnetic elements: 0.0 muB per atom
Omitting ISPIN = 2 for magnetic systems will give incorrect energies,
wrong magnetic ground states, and unreliable forces.

## implementation

**CRITICAL: INCAR generation rules.** Always follow these:

1. ALWAYS include GGA tag explicitly (usually GGA = PE for PBE).
2. Set ENCUT >= 400 eV for systems containing hydrogen. The H POTCAR
   has a low default ENCUT, but accurate H binding energies require
   at least 400 eV. 450 eV is standard practice.
3. Use EDIFF = 1E-6 for production calculations. 1E-4 is acceptable
   only for initial rough relaxations.
4. Set EDIFFG = -0.01 to -0.03 eV/Ang for force-based convergence
   in relaxations. Negative values specify force convergence (preferred),
   positive values specify energy convergence.

**Surface slab calculations:**
- Use ISIF = 2 (relax ions, fix cell shape and volume). ISIF = 3 will
  relax the vacuum, collapsing the slab.
- K-points should be 1 in the direction perpendicular to the slab
  (the vacuum direction, usually z).
- Consider dipole corrections for asymmetric slabs: IDIPOL = 3,
  LDIPOL = .TRUE.
- Use selective dynamics to fix bottom layers of the slab. At least
  half the slab layers should be fixed to represent bulk.

**Bulk calculations:**
- For equation of state or cell optimization: ISIF = 3 (relax everything)
  or ISIF = 7 (relax volume only).
- K-point density should give convergence to less than 1 meV/atom. Typical
  minimum: approximately 0.03 inverse Angstrom spacing in reciprocal space.

**Charge analysis (Bader):**
- Set LCHARG = .TRUE. and LAECHG = .TRUE. to write all-electron charge
  density files (AECCAR0, AECCAR2) needed for Bader analysis.
- NGXF, NGYF, NGZF can be increased for finer charge density grids
  (typically 2x the default FFT grid).

**NEB calculations:**
- IBRION = 3 (damped MD, required for VTST NEB)
- POTIM = 0 (for VTST tools) or a small value
- IMAGES = N (number of intermediate images, typically 4-8)
- LCLIMB = .TRUE. (climbing image NEB for accurate barrier)
- SPRING = -5 (negative value for nudged elastic band)
- NSW should be large enough for convergence (200-500)

**Parallelization:**
- NCORE = number of cores per orbital band (typically 4-8)
- KPAR = number of k-point groups (divide total cores by NCORE to
  estimate; must evenly divide the number of k-points)
- Do not set both NCORE and NPAR (they are reciprocals)
- For large systems (more than 200 atoms): consider LREAL = Auto
- For small systems (fewer than 20 atoms): LREAL = .FALSE. is safer

**K-points rules:**
- Bulk metals: Gamma-centered Monkhorst-Pack grid, density approximately 0.03 inverse Angstrom
- Surface slabs: Same in-plane density, 1 in vacuum direction
- Molecules in a box: Gamma point only (1 1 1)
- Hexagonal systems: Always use Gamma-centered grids (not standard MP)
- NEB: Usually reduce k-points vs single-point for computational cost

**INCAR template for metal surface relaxation:**

  GGA = PE
  ENCUT = 450
  PREC = Accurate
  EDIFF = 1E-6
  ISMEAR = 1
  SIGMA = 0.1
  LREAL = Auto
  ALGO = Normal
  IBRION = 2
  ISIF = 2
  NSW = 200
  EDIFFG = -0.01
  LORBIT = 11
  LCHARG = .TRUE.
  LWAVE = .FALSE.
  NCORE = 4
  KPAR = 2

**INCAR template for bulk equation of state:**

  GGA = PE
  ENCUT = 450
  PREC = Accurate
  EDIFF = 1E-6
  ISMEAR = 1
  SIGMA = 0.1
  ALGO = Normal
  IBRION = 2
  ISIF = 3
  NSW = 200
  EDIFFG = -0.01
  LORBIT = 11
  LWAVE = .FALSE.
  NCORE = 4
  KPAR = 2

**INCAR template for molecule in vacuum:**

  GGA = PE
  ENCUT = 450
  PREC = Accurate
  EDIFF = 1E-6
  ISMEAR = 0
  SIGMA = 0.01
  LREAL = .FALSE.
  ALGO = Normal
  IBRION = 2
  ISIF = 2
  NSW = 100
  EDIFFG = -0.01
  LWAVE = .FALSE.
  NCORE = 1
  KPAR = 1

## validation

**Pre-submit syntax check (engine-native, no LLM):**

VASP accepts unknown INCAR keys silently — a one-letter typo such as
`ISPN = 2` instead of `ISPIN = 2` disables spin polarisation and
produces a physics-wrong result that converges by every other metric.
Before submission, `PeriodicDFTAgent` runs the generated INCAR through
`scilink.agents.sim_agents.vasp_input_validator.check_incar_syntax`
(pymatgen's `Incar.check_params()`). High-confidence typos are
auto-renamed to the closest valid tag; low-confidence matches are
returned for downstream LLM review. The fix payload is recorded under
`result["syntax_check"]` so the caller can log what was changed.

This is the VASP instance of an engine-neutral contract:
`<engine>_input_validator.check_syntax(content) -> List[issue]`. Don't
add tag-spelling guidance to LLM prompts — the pymatgen check is the
canonical authority, and the LLM should reason about physics, not
syntax.

**Quality checks for generated INCAR files:**

- GGA tag MUST be present and match the intended functional. Its absence
  is the single most common cause of POTCAR lookup failures.
- ENCUT must be at least 1.3x the maximum ENMAX in any POTCAR used. For
  systems with H, this means at least 400 eV minimum.
- ISMEAR must match the system type. Metals: 1 or 2 (MP) for both
  static and relaxation calcs -- do not switch to -5 just because
  NSW = 0. Molecules and semiconductor relaxations: 0 (Gaussian).
  Static (NSW = 0) insulators: -5 (tetrahedron) preferred for
  accurate energies, 0 also acceptable.
- ISIF must match the calculation type: 2 for slabs, 3 for bulk
  cell optimization, 0 for fixed-cell single points.
- ISPIN = 2 must be present for any system containing Ni, Fe, Co, Mn, Cr.
  MAGMOM must also be set with chemically reasonable initial values.
- For slabs: k-points must be 1 in the vacuum direction.
- For molecules: k-points must be 1 1 1 (Gamma only).
- NSW must be greater than 0 for relaxations, NSW = 0 for single-point calculations.
- LCHARG and LAECHG must be .TRUE. if Bader analysis is requested.
- NCORE and KPAR should be set for efficient parallelization.
- Check for contradictory settings: for example ISMEAR = -5 with NSW > 0,
  or ISIF = 3 with a slab geometry.

**Common errors and their INCAR fixes:**

- "No pseudopotential for X": Missing or wrong GGA tag. Fix: add GGA = PE.
- "ZBRENT: fatal error": POTIM too large. Fix: reduce POTIM to 0.1, use IBRION = 1.
- "Sub-Space-Matrix not hermitian": ALGO incompatibility. Fix: use ALGO = Normal.
- "BRMIX: very serious problems": SCF not converging. Fix: add AMIX=0.1, BMIX=0.01.
- "RSPHER": Real-space projection overlap. Fix: set LREAL = .FALSE.
- "Highest band is occupied": Not enough bands. Fix: increase NBANDS.
- "EDDDAV did not converge": Electronic convergence failure. Fix: ALGO = All, increase NELM.
