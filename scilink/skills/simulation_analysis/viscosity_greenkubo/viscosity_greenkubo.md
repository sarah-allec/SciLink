---
description: Shear viscosity via the Green-Kubo stress-autocorrelation integral from a logged pressure-tensor time series.
technique: green_kubo
computes: [shear_viscosity]
requires: [thermo_log]
---

## Overview

Shear viscosity η from equilibrium MD via the Green-Kubo relation — the time
integral of the off-diagonal stress (pressure-tensor) autocorrelation. It needs
a densely and long-enough sampled time series of the pressure tensor from an
equilibrated NVT/NPT production run; it does not use the coordinate trajectory.
The estimate is statistically noisy and sensitive to run length and sampling
cadence, so convergence must be checked, not assumed.

## Implementation

Read the pressure-tensor time series and its timestep from the thermo/stress log,
then evaluate the Green-Kubo integral:

1. **Parse** the off-diagonal pressure components Pxy, Pxz, Pyz as a time series
   with the sampling interval dt. **Identify the columns from the header,
   matching `pxy`/`pxz`/`pyz` as a case-insensitive substring** — a `fix
   ave/time` file writes a **two-line** comment header and labels these columns
   `v_pxy v_pxz v_pyz` (the LAMMPS `v_` variable prefix) on the *second* `#` line,
   e.g. `# TimeStep v_pxy v_pxz v_pyz v_temp v_vol`; a `thermo_style custom … pxy
   pxz pyz` log names them `pxy pxz pyz`. **Scan every leading `#` line, not just
   the first** (the first `fix ave/time` line is just a title). Map data columns
   by position from that label line. Also read the volume V and temperature T —
   the `v_vol` and `v_temp` columns here — or from the run metadata.

2. **Autocorrelation.** For each independent stress component P_k(t), compute the
   autocorrelation C_k(τ) = ⟨P_k(t) P_k(t+τ)⟩ averaged over time origins. Use an
   FFT-based estimator and normalize by the number of origins at each lag (NOT by
   C(0) — the absolute magnitude carries the units).

3. **Average equivalent components** to cut noise: the three off-diagonal terms
   (Pxy, Pxz, Pyz) plus the two independent traceless diagonal combinations
   (Pxx−Pyy)/2 and (Pyy−Pzz)/2. All five are unbiased estimators of the same shear
   viscosity; average their autocorrelations.

4. **Green-Kubo integral:** η = (V / (k_B T)) ∫₀^∞ C(τ) dτ. Integrate with the
   trapezoidal rule and track the *running* integral; take η as the plateau value.
   If the running integral is still rising at the end of the series, it has not
   converged — report the best estimate and set plateau_reached=false.

5. **Units — do this explicitly and state the assumed unit system.** LAMMPS
   `real`: pressure in atm, time in fs, volume in Å³, T in K. LAMMPS `metal`:
   pressure in bar, time in ps, volume in Å³. Convert the final η to Pa·s, then
   report in mPa·s. A missing/incorrect unit conversion is the most common error
   and shows up as a value orders of magnitude off.

6. **Pool independent replicas when present.** Green-Kubo is noisy, so a
   trustworthy value comes from averaging independent runs that differ only in
   their initial velocity seed. **Discover replicas from DATA_FILES by content,
   not by filename** — each run names its own stress file:
   - From DATA_FILES, keep every file whose header contains the off-diagonal
     pressure columns, **matching `pxy`/`pxz`/`pyz` as a case-insensitive
     substring across ALL leading `#` header lines** — so a `fix ave/time` file
     labeling them `v_pxy v_pxz v_pyz` on its second comment line qualifies (see
     step 1). That substring test *is* how you find the stress series — discard
     plain thermo logs (`log.lammps`, `run_stdout.log`) that lack those columns,
     instead of matching any particular filename.
   - **Skip dry-run artifacts:** ignore any path with a component containing
     `dryrun` (e.g. `.../_dryrun/…`) — short setup checks, not production.
   - **One replica per parent directory:** group the kept files by parent
     directory, so replicas in sibling subdirectories named *anything*
     (`velocityseed_12345/`, `member_0/`, `rep_1/`, …) are each counted once; if
     a directory has more than one qualifying file, keep a single stress series.
   Compute η for each replica by steps 1–5, then report the **mean across
   replicas** as `value` and the **standard error of the mean** as `std_error`.
   One qualifying file is a single replica (no pooling, `std_error` null). If the
   header test finds no pressure-tensor file at all, fail with a message naming
   what was searched. Base `plateau_reached` on both signals: each replica's
   running integral should plateau, AND the inter-replica spread should be small
   relative to the mean (a large spread means more replicas or longer runs are
   needed, so `plateau_reached=false` even if individual integrals look flat).

Print one JSON object as the last stdout line:
`{"status":"success","value":<mean η in mPa·s>,"units":"mPa·s","plateau_reached":<bool>,"n_replicas":<int>,"std_error":<mPa·s or null>,"n_origins":<int>}`.
With a single log, `n_replicas` is 1 and `std_error` is null. On failure:
`{"status":"error","message":<str>}`.

## Validation

Green-Kubo viscosity is noisy; guard against false precision:
- The autocorrelation must decay to ≈0 well before the integration cutoff.
- The running integral must show a plateau; a still-rising integral is not
  converged (plateau_reached=false).
- Typical liquid viscosities are ~0.1–10 mPa·s; a value orders of magnitude
  outside this range signals a unit error, not physics.
- With replicas, report the standard error, not just the mean: a mean whose
  standard error is a large fraction of it is not yet converged, regardless of
  how flat any single replica's integral looks.

## Interpretation

η is the shear viscosity. Non-polarizable water models (notably TIP3P)
characteristically *underestimate* viscosity, so a low value can reflect the
force field rather than the analysis. Always report the value with its
convergence status so the downstream validation panel can judge it against
measured data rather than trusting it blind.
