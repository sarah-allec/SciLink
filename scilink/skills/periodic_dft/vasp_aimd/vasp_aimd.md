---
description: VASP ab initio molecular dynamics (AIMD) input generation — Born-Oppenheimer MD via IBRION=0, thermostat selection (MDALGO/SMASS/LANGEVIN_GAMMA), timestep (POTIM) and run-length (NSW) choices, and the electronic-structure settings that make MD stable and affordable. For NVT/NPT dynamics of liquids, solutions, and solvated ions.
# Engine-native coordinate file: VASP reads the structure from a separate POSCAR.
# The agent writes it deterministically (via ASE, format below) from the generated
# coordinates, so the LLM never transcribes positions and the file is guaranteed.
structure_file: POSCAR
structure_format: vasp
detect:
  binaries: [vasp_std, vasp_gam, vasp_ncl, vasp]
  env_vars: [VASP_HOME, VASP_DIR]
  python_modules: []
  guidance: |
    VASP ships several binaries flavored by k-point sampling:
    vasp_std (general k-points), vasp_gam (gamma-only, fastest for
    large cells), vasp_ncl (noncollinear / spin-orbit). AIMD cells are
    large and sampled at Gamma only, so vasp_gam is usually the right
    (and fastest) binary. Some sites install them under
    $VASP_HOME/<flavor> or $VASP_DIR/bin/; on HPC with Lmod the binary
    may only be on $PATH after `module load vasp/<version>`. Detection
    should consider any of the above as a positive hit.
---
# VASP Ab Initio Molecular Dynamics (AIMD) Input Generation Skill

## overview

Born-Oppenheimer molecular dynamics with VASP: the forces come from a
full self-consistent DFT solve at every timestep, and the ions are
propagated with those forces. This skill covers the INCAR settings that
make AIMD both *stable* (correct thermostat, timestep, and electronic
convergence) and *affordable* (Gamma-only sampling, a single relaxation
per step, no wasted accuracy). It targets finite-temperature dynamics of
condensed-phase systems — liquids, solutions, and solvated ions in a
periodic box — not static energies or 0 K relaxations.

The distinguishing INCAR tag is **IBRION = 0** (molecular dynamics). This
is a fundamentally different calculation from a relaxation (IBRION = 1/2)
or a single point (NSW = 0): the goal is a physically faithful *trajectory*
at temperature, from which properties (structure, coordination, dynamics,
and per-step energies/forces) are harvested — not a single minimized
geometry. Do NOT treat an AIMD request as a relaxation.

## planning

**Confirm it is really MD.** AIMD is requested when the goal is dynamics
at a temperature (sampling, diffusion, coordination, speciation, solvation
structure, or generating configurations/forces for MLIP training). If the
goal is a minimum-energy geometry, that is a relaxation (use the `vasp`
skill), not this one.

**Ensemble and thermostat.** Pick the ensemble from what is held fixed:
- **NVT** (fixed volume, controlled temperature) — the default for a
  liquid/solution already at a sensible density. Thermostat options:
  - Nose-Hoover (MDALGO = 2): set SMASS >= 0 (a small positive value, e.g.
    SMASS = 0 to ~3, controls the thermostat coupling period). Deterministic,
    good for equilibrium properties.
  - Langevin (MDALGO = 3): set LANGEVIN_GAMMA (friction, one value per
    species type, in ps^-1; ~1-10 is typical). Robust for equilibration and
    for strongly coupled/ionic liquids. Requires ISIF = 2.
- **NPT** (fixed pressure) — Langevin barostat (MDALGO = 3) with
  LANGEVIN_GAMMA_L and PMASS set, ISIF = 3. Use only when the density must
  relax; for a pre-packed box at target density, prefer NVT and skip
  barostat complexity.

**Temperature.** Set TEBEG (and TEEND = TEBEG for a constant-temperature
run). Aqueous systems: 300-350 K. A brief higher-temperature start can help
melt an artificial packing, but keep the production segment at the target T.

**Timestep (POTIM, in fs).** The timestep must resolve the fastest motion.
Systems containing hydrogen (water, hydroxide, any O-H/N-H) require
POTIM <= 1.0 fs (0.5 fs is safest). Without light atoms, 1-2 fs is
acceptable. Too large a POTIM makes the dynamics blow up or the temperature
drift upward — this is the most common AIMD failure.

**Run length (NSW).** AIMD needs enough steps to equilibrate and then
sample. NSW is large by nature: hundreds to a few thousand steps. Report
the physical time (NSW * POTIM). For a small demonstration cell, a few
hundred steps equilibration + a production segment is typical; note that
expense scales with NSW * (cost of one SCF).

## implementation

**CRITICAL: AIMD INCAR generation rules.** Always follow these:

1. **IBRION = 0** — this is what makes it MD. Never IBRION = 1/2/3 for AIMD.
2. **Set a thermostat** — MDALGO = 2 (Nose-Hoover, needs SMASS) or MDALGO = 3
   (Langevin, needs LANGEVIN_GAMMA per species). An MD run without a
   thermostat drifts; do not omit it for a controlled-temperature run.
3. **POTIM <= 1.0** for any system with hydrogen (0.5 fs safest). This is
   non-negotiable for aqueous/hydroxide systems.
4. **TEBEG (and TEEND)** must be set to the target temperature.
5. **ISMEAR = 0** (Gaussian) with a small SIGMA (0.05-0.1 eV). NEVER
   ISMEAR = -5 for MD — tetrahedron gives erratic forces and will not run
   stable dynamics.
6. **ALWAYS include the GGA tag** (GGA = PE for PBE) — same POTCAR-directory
   rule as static VASP; omitting it causes a fatal "No pseudopotential" error.
7. **ENCUT >= 400 eV** for hydrogen-containing systems (450 eV standard).
8. **EDIFF = 1E-5** is the practical AIMD setting: MD needs consistent forces
   every step, but 1E-6 (static-quality) makes every one of thousands of
   steps needlessly expensive. 1E-5 balances force quality against cost.
9. **NELMIN = 4-6** — force at least a few SCF cycles per step so forces stay
   consistent as the charge density is re-converged from the previous step.

**Electronic settings for efficient MD:**
- ALGO = Normal (blocked Davidson) or VeryFast (RMM-DIIS) — MD reuses the
  previous step's wavefunction, so per-step SCF is cheap once warmed up.
- LREAL = Auto for large cells (condensed-phase boxes are big); LREAL = .FALSE.
  only for small cells (< ~20 atoms).
- LWAVE = .FALSE. and LCHARG = .FALSE. — do NOT write WAVECAR/CHGCAR every
  run by default; they are large and not needed for a trajectory. (Enable
  only if restarting.)
- MAXMIX = 40-60 can speed charge-density remixing between MD steps.
- ISYM = 0 — turn off symmetry for MD; the moving ions break it and symmetry
  reduction can cause errors.

**Writing the trajectory and forces (needed downstream):**
- VASP writes the trajectory to XDATCAR (positions per step) and full
  per-step energies/forces to OUTCAR and vasprun.xml automatically. To
  harvest (configuration, energy, forces) tuples for MLIP training, ensure
  vasprun.xml is written (it is, by default) — no special tag is required,
  but do NOT set LWAVE/LCHARG just to capture dynamics.
- NBLOCK = 1 writes every step to XDATCAR (default); raise it only to thin a
  very long trajectory.

**Spin polarization:** systems with Ni, Fe, Co, Mn, Cr still require
ISPIN = 2 with MAGMOM, exactly as in static VASP. (An aqueous alkaline-earth
system — Mg, Ca, Na, O, H — is non-magnetic: ISPIN = 1.)

**K-points:** condensed-phase AIMD cells are large, so **Gamma point only
(1 1 1)** is correct and standard. A larger k-mesh multiplies an already
expensive per-step SCF for negligible gain at these cell sizes.

**INCAR template for NVT AIMD of an aqueous/solvated system (Nose-Hoover):**

  GGA = PE
  ENCUT = 450
  PREC = Normal
  EDIFF = 1E-5
  ISMEAR = 0
  SIGMA = 0.05
  ISPIN = 1
  ALGO = Normal
  LREAL = Auto
  ISYM = 0
  IBRION = 0
  MDALGO = 2
  SMASS = 0
  POTIM = 0.5
  NSW = 2000
  TEBEG = 300
  TEEND = 300
  NELMIN = 4
  LWAVE = .FALSE.
  LCHARG = .FALSE.
  NCORE = 4
  KPAR = 1

**INCAR template for NVT AIMD (Langevin thermostat, robust for ionic liquids):**

  GGA = PE
  ENCUT = 450
  PREC = Normal
  EDIFF = 1E-5
  ISMEAR = 0
  SIGMA = 0.05
  ISPIN = 1
  ALGO = VeryFast
  LREAL = Auto
  ISYM = 0
  IBRION = 0
  MDALGO = 3
  LANGEVIN_GAMMA = 10.0 10.0 10.0 10.0 10.0
  ISIF = 2
  POTIM = 0.5
  NSW = 2000
  TEBEG = 300
  TEEND = 300
  NELMIN = 4
  LWAVE = .FALSE.
  LCHARG = .FALSE.
  NCORE = 4
  KPAR = 1

  # NOTE: LANGEVIN_GAMMA needs one friction value per species TYPE, in the
  # POSCAR's species order. The example lists five (e.g. O H Mg Ca Na); match
  # the count and order to the actual structure.

**KPOINTS (Gamma only):**

  Gamma-only AIMD
  0
  Gamma
  1 1 1
  0 0 0

## interpretation

Read a finished AIMD run as a *trajectory*, not a single result — the
questions are different from a relaxation.

**Temperature stability is the first health check.** The running-average
temperature should sit near TEBEG without a systematic upward drift. A
steady climb almost always means POTIM is too large (energy is being
injected because the integrator cannot resolve the fastest vibration) —
reduce it (halve it for hydrogen systems) before trusting anything else.
A thermostat that never settles suggests SMASS/LANGEVIN_GAMMA coupling is
mistuned.

**Distinguish equilibration from production.** Early steps carry the memory
of the artificial starting packing; properties must be measured only after
the system has relaxed to its thermal ensemble (temperature, energy, and
structural measures have plateaued). Reporting an average over the whole
run, including the transient, is a common error.

**Energy conservation / drift.** Even with a thermostat, a large monotonic
drift in the conserved quantity signals too large a timestep or too loose an
EDIFF (inconsistent forces step to step). Tighten EDIFF toward 1E-5/1E-6 or
reduce POTIM.

**Per-step forces feed MLIP training.** When the run's purpose is training
data, the forces in vasprun.xml/OUTCAR are the product — verify they are
finite and physically scaled (no runaway values from an unstable step) before
harvesting.

**Error-pattern triage (MD-specific; the static `vasp` skill covers the
general SCF failures):**

- *Temperature drifts upward / dynamics "explode"* — POTIM too large for the
  fastest motion. Reduce POTIM (0.5 fs for H-containing systems); confirm a
  thermostat is set.
- *`ZBRENT` / ionic instability at start* — the initial packing has close
  contacts. Run a short relaxation or a brief low-T equilibration segment
  first, or reduce POTIM for the opening steps.
- *SCF not converging within NELM mid-trajectory* — the reused wavefunction
  went stale after a large ionic move; raise NELM, ensure NELMIN >= 4, or use
  ALGO = Normal (more robust than VeryFast) if VeryFast stalls.
- *Run far too slow* — check that k-points are Gamma-only, LWAVE/LCHARG are
  .FALSE., EDIFF is 1E-5 (not 1E-6), and PREC = Normal (not Accurate); AIMD
  does not need static-quality settings every step.

## validation

**Pre-submit syntax check (engine-native, no LLM):** identical contract to
the static `vasp` skill — `PeriodicDFTAgent` runs the generated INCAR through
`scilink.agents.sim_agents.vasp_input_validator.check_incar_syntax`
(pymatgen's `Incar.check_params()`), which catches silently-ignored tag
typos. Do not add tag-spelling guidance to prompts; reason about physics.

**Quality checks specific to an AIMD INCAR:**

- **IBRION = 0 must be present.** Its absence means the run is not MD. IBRION
  = 1/2/3 with an AIMD request is wrong.
- **A thermostat must be set** for a controlled-temperature run: MDALGO = 2
  with SMASS, or MDALGO = 3 with LANGEVIN_GAMMA. Flag an IBRION = 0 run with
  no thermostat.
- **POTIM must be <= 1.0** (ideally 0.5) whenever the structure contains
  hydrogen. A larger timestep with H present is a substantive error.
- **TEBEG must be set** (and TEEND, = TEBEG for constant T).
- **ISMEAR must be 0** (Gaussian) for MD. ISMEAR = -5 with IBRION = 0 is a
  contradiction — flag it.
- **NSW must be large** (>> 1; hundreds to thousands). NSW = 0 or a handful
  contradicts an MD request.
- **GGA tag present** (POTCAR-directory correctness), and **ENCUT >= 400 eV**
  for hydrogen-containing systems — same as static VASP.
- **K-points Gamma-only (1 1 1)** for a condensed-phase cell; a dense mesh is
  a (very expensive) mistake here.
- **LANGEVIN_GAMMA count matches the number of species types** when MDALGO = 3.
- **ISPIN = 2 with MAGMOM** only if magnetic elements are present (not the
  case for a Mg/Ca/Na/O/H aqueous system).
- Check for contradictions: ISMEAR = -5 with IBRION = 0; a thermostat tag set
  while IBRION != 0; POTIM > 1 with hydrogen present; NSW = 0 with IBRION = 0.

Post-run trajectory diagnosis lives in the `interpretation` section; the
checks here apply to the INCAR before submission.
