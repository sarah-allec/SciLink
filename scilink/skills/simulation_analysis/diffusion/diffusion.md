---
description: Self-diffusion coefficient from a trajectory via the Einstein mean-squared-displacement relation, read engine-agnostically.
technique: einstein_msd
computes: [self_diffusion]
requires: [trajectory]
---

## Overview

The self-diffusion coefficient D of a species from equilibrium MD, via the
Einstein relation: D is one sixth the long-time slope of the mean-squared
displacement (MSD). It works on any coordinate trajectory — LAMMPS, ASE/MLIP,
GROMACS — because the trajectory is read with MDAnalysis/ASE, which handle the
formats. The same skill therefore serves classical MD and MLIP-driven MD.

## Implementation

1. **Read the trajectory** from DATA_FILES with MDAnalysis (`mda.Universe`) or
   ASE (`ase.io.read(..., index=":")`) — do not name an engine. Identify the
   diffusing species from the research goal; record the timestep dt between
   frames. **Unwrap** periodic coordinates (or use unwrapped positions) so the
   displacement is not reset at box boundaries.

2. **Mean-squared displacement:** MSD(τ) = ⟨ |r_i(t+τ) − r_i(t)|² ⟩, averaged
   over particles i of the species and over time origins t. Use an FFT-based MSD
   estimator for efficiency; report the number of origins.

3. **Einstein fit:** in the diffusive (long-time) regime MSD is linear in τ with
   slope 6D (3-D). Fit a straight line to the linear portion — skip the early
   ballistic/sub-diffusive regime and the noisy tail (roughly the middle
   third–half of the MSD curve) — and take D = slope / 6. Flag if no linear
   regime is found (the run is too short or sub-diffusive).

4. **Units:** MSD in Å², τ in ps (or fs) → D in Å²/ps; convert to cm²/s
   (1 Å²/ps = 1e-4 cm²/s) and state the conversion. Print one JSON object as the
   last stdout line: `{"status":"success","value":<D in cm^2/s>,"units":"cm^2/s",
   "n_origins":<int>,"linear_regime":<bool>}`. On failure:
   `{"status":"error","message":<str>}`.

## Validation

- The MSD must have a **linear regime**; fitting the ballistic short-time part or
  the noisy tail gives a wrong D. A curve that is still sub-diffusive at the end
  means the run is too short — set `linear_regime` false.
- D must be **positive**. Typical liquid self-diffusion is ~0.5–5 ×10⁻⁵ cm²/s
  (bulk water ≈ 2.3×10⁻⁵ cm²/s at 298 K); values orders of magnitude off signal a
  unit error or unwrapping failure.
- Forgetting to unwrap periodic coordinates caps the MSD at the box size and
  drastically underestimates D — check that the MSD grows without bound.

## Interpretation

D reports the translational mobility of the species in its environment. Slower
diffusion (smaller D) reflects stronger interactions / higher viscosity. Report D
with its convergence status so a trend across state points or compositions is
judged against measurement, not read as exact.
