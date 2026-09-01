---
description: Self-diffusion coefficient D from the mean-squared displacement (MSD) via the Einstein relation, from an equilibrium MD trajectory.
technique: einstein_msd
computes: [self_diffusion]
requires: [trajectory]
---

## Overview

Self-diffusion coefficient D from equilibrium MD via the Einstein relation: the
mean-squared displacement MSD(t) = ⟨|r_i(t0+t) − r_i(t0)|²⟩ grows linearly in the
diffusive (Fickian) regime, and D = MSD_slope / (2·d) with d the number of spatial
dimensions the displacement spans (d = 3 for an isotropic bulk liquid → D =
slope/6). It needs the coordinate **trajectory**; the positions MUST be unwrapped
across periodic boundaries first (a wrapped MSD saturates at the box size and gives
a meaningless D). D converges far more reliably than a Green–Kubo transport
coefficient — it is a single-particle average over every diffusing atom, so short
trajectories already give a well-defined slope.

## Implementation

1. **Load** the trajectory (MDAnalysis) and read the box and the sampling interval
   dt from the dump. Select the diffusing species (all atoms, or a molecular
   subset — see Interpretation).
   **STRIDE a large trajectory — do this before anything else.** A sub-picosecond
   dump over many ns is enormous (often >1e4 frames, many GB) — far finer in time
   than a diffusion MSD needs, and too big to parse in one pass or hold in memory
   (the position array is N_atoms × N_frames × 3 floats). Read only every Nth frame
   (e.g. `MDAnalysis.Universe(...).trajectory[::stride]`, or the reader's `step=`),
   choosing the stride so **~2000–5000 frames** remain, which gives ps-to-ns
   spacing. Subsampling in time does NOT bias the slope — the MSD is smooth — it
   only coarsens the lag grid. Multiply the effective dt by the stride. Never
   materialize the full trajectory at once.

2. **Find the coordinate columns FIRST — do not assume `x y z`.** Read the
   `ITEM: ATOMS` header and handle whichever coordinate columns the run actually
   wrote. A deck may dump wrapped OR unwrapped positions under different names; a
   parser that only looks for `x y z` will wrongly report "no coordinate columns"
   on a perfectly good unwrapped dump. Accept, in this order of preference:
   - **Unwrapped `xu yu zu` (preferred — use directly, no unwrapping).** These are
     LAMMPS *unwrapped* coordinates: exactly the MSD input you want, already
     continuous across periodic boundaries. Read them as the positions and SKIP the
     unwrapping steps below. (`xsu ysu zsu` are scaled-unwrapped — the same once
     multiplied by the box vectors.)
   - **Wrapped `x y z` + image flags `ix iy iz`.** Reconstruct the true position
     `r_unwrapped = r_wrapped + (ix, iy, iz) · (Lx, Ly, Lz)`, per atom per frame.
   - **Wrapped `x y z`, no image flags.** Displacement-unwrap: accumulate
     displacements between consecutive frames, subtracting one box vector per
     dimension whenever a coordinate jumps by more than L/2 (a PBC crossing). Valid
     only when sampling is frequent enough that no atom moves more than L/2 per
     frame — true for a sub-ps dump. (Scaled-wrapped `xs ys zs` must be multiplied
     by the box first.)
   **UNWRAPPED coordinates are mandatory for the MSD** — wrapped positions saturate
   at the box size and give a meaningless D. Sanity check: an unwrapped MSD keeps
   growing; a still-wrapped one saturates near (L/2)². If you see saturation, the
   unwrap did not take.

3. **MSD.** Compute MSD(t) averaged over atoms AND time origins. Use the FFT-based
   estimator (MDAnalysis `analysis.msd.EinsteinMSD` with `fft=True`, on
   already-unwrapped positions, or a Fourier autocorrelation). The naïve double
   loop over origins is O(N_frames²) and far too slow for a long trajectory.

4. **Fit the diffusive (Fickian) regime, not the whole curve.** The MSD has a
   short-time BALLISTIC region (log–log slope ≈ 2) and a noisy long-time tail (few
   origins). Find the middle window where the **log–log slope ≈ 1** and fit
   `MSD = 2·d·D·t + c` there by linear regression. Report the fit R² and the
   log–log slope of that window (should be ~0.9–1.1).

5. **Units.** Convert to m²/s. LAMMPS `real`: positions in Å, time in fs, so
   `D[Å²/fs] × 1e-5 = D[m²/s]`. LAMMPS `metal`: Å and ps → `× 1e-8`. Report in m²/s
   (bulk liquids are ~1e-9 m²/s).

Print one JSON object as the last stdout line:
`{"status":"success","value":<D in m^2/s>,"units":"m^2/s","fickian":<bool>,"r2":<float>,"loglog_slope":<float>}`.
On failure: `{"status":"error","message":<str>}`.

## Validation

- The MSD MUST show a clear linear (Fickian) region — log–log slope ≈ 1 — before
  fitting. A still-curving (sub-diffusive) MSD means D is not yet converged: set
  `fickian=false` and do NOT report a slope from a non-linear region.
- Unwrap sanity: an MSD that saturates near (L/2)² means the coordinates were NOT
  unwrapped — the result is invalid, not a small D.
- Bulk-liquid self-diffusion is ~1e-11 to 1e-9 m²/s; a value orders of magnitude
  outside signals a unit or unwrap error, not physics.

## Interpretation

D is the self-diffusion coefficient. It falls as viscosity rises (Stokes–Einstein,
D ∝ T/η), so a diffusion **trend** across a swept parameter is a robust transport
signal that resolves where a Green–Kubo viscosity does not. For a trajectory of a
molecular SUBSET (e.g. only the water hydrogens), the long-time MSD slope gives the
MOLECULE's translational diffusion — the fast intramolecular motion (rotation,
vibration) contributes only a bounded offset at short times and drops out of the
slope. Report D with its Fickian/convergence status so the downstream validation
panel can judge it against measured data (e.g. PFG-NMR self-diffusion).
