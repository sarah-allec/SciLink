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
   with the sampling interval dt. Identify the columns by header — they appear in
   a `thermo_style custom … pxy pxz pyz` log or a `fix ave/time` output file. Also
   read the volume V and temperature T (from the log or the run metadata).

2. **Autocorrelation.** For each independent stress component P_k(t), compute the
   autocorrelation C_k(τ) = ⟨P_k(t) P_k(t+τ)⟩ averaged over time origins. Use an
   FFT-based estimator and normalize by the number of origins at each lag (NOT by
   C(0) — the absolute magnitude carries the units).

3. **Average equivalent components** to cut noise: the three off-diagonal terms
   (Pxy, Pxz, Pyz) plus the two independent traceless diagonal combinations
   (Pxx−Pyy)/2 and (Pyy−Pzz)/2. All five are unbiased estimators of the same shear
   viscosity; average their autocorrelations.

4. **Green-Kubo integral — extract the plateau by fitting, never a raw endpoint.**
   Form the *running* integral η(t) = (V / (k_B T)) ∫₀^t C(τ) dτ by cumulative
   trapezoidal integration. Do NOT report η at the final lag: the running
   integral's tail has a variance that grows with t (every added lag injects
   noise), so the endpoint is unstable and can even go *negative* — a spurious
   result, not a real viscosity. Extract the plateau robustly instead:
   - **Fit cutoff.** Track the spread of the running integral across the five
     equivalent components; set a cutoff `t_cut` where that inter-component
     relative spread first exceeds ≈0.4 (beyond it the tail is noise-dominated).
   - **Fit** the averaged running integral on `[0, t_cut]` to the standard
     double-exponential rise-to-plateau
     `η(t) = A[ α·τ1·(1−e^(−t/τ1)) + (1−α)·τ2·(1−e^(−t/τ2)) ]`,
     weighting each point by `t^(−b)` (b ≈ 0.5–1) so the noisy long-time tail does
     not dominate (Zhang, Otani & Maginn, *J. Chem. Theory Comput.* 2015). The
     fitted amplitude **A** (the t→∞ limit) is η.
   - **Convergence + sanity.** `plateau_reached` is true only when the fit
     succeeds AND `A > 0` AND the fit is stable over the upper part of the window.
     A negative or non-finite A, a failed fit, or an integral still rising at
     `t_cut` means NOT converged: report the best estimate with
     `plateau_reached=false` — never emit a negative or wild value as if real.
   - Also report a per-run uncertainty (spread of A across the equivalent
     components, or across time-origin blocks).

5. **Units — do this explicitly and state the assumed unit system.** LAMMPS
   `real`: pressure in atm, time in fs, volume in Å³, T in K. LAMMPS `metal`:
   pressure in bar, time in ps, volume in Å³. Convert the final η to Pa·s, then
   report in mPa·s. A missing/incorrect unit conversion is the most common error
   and shows up as a value orders of magnitude off.

Print one JSON object as the last stdout line:
`{"status":"success","value":<η in mPa·s>,"units":"mPa·s","plateau_reached":<bool>,"n_origins":<int>}`.
On failure: `{"status":"error","message":<str>}`.

### Convergence over replicas — let the data set the sample size

A single trajectory's Green-Kubo η is too noisy to trust even when its own
running integral plateaus: the *estimate of the mean* only settles across several
INDEPENDENT replicas (different seeds/initial velocities). Do not fix the replica
count up front — converge adaptively with `run_convergence_loop`
(`scilink.agents.sim_agents.convergence`): it runs replicas, measures η per
replica with this skill, and stops when SciLink judges the running mean settled.

Criterion (the default judge): at least `min_replicas` (≥3), **every** replica's
running integral plateaued, and the **relative standard error of the mean**
(`std/√n / |mean|`) at or below the target (~0.10, i.e. ±10%). If the cap
(`max_replicas`) is hit first, report the best mean with `converged=false` — it
stays `[UNVERIFIED]` and loops back for more sampling rather than being trusted.
Report the converged mean **with its uncertainty** (relative SEM); a viscous
mixture has long stress-correlation times, so it needs longer per-replica runs
and/or more replicas than a mobile one — that is exactly what the adaptive loop
decides.

## Validation

Green-Kubo viscosity is noisy; guard against false precision:
- The autocorrelation must decay to ≈0 well before the integration cutoff.
- η comes from the fitted plateau amplitude A, not the raw endpoint of the running
  integral; a fitted A that is negative or non-finite is NOT a viscosity — report
  plateau_reached=false, never a negative or wild value.
- The running integral must show a plateau; a still-rising integral is not
  converged (plateau_reached=false).
- The inter-replica mean must be converged: relative SEM ≤ target with every
  replica plateaued, else the value is not yet verified (add replicas).
- Typical liquid viscosities are ~0.1–10 mPa·s; a value orders of magnitude
  outside this range signals a unit error, not physics.

## Interpretation

η is the shear viscosity. Non-polarizable water models (notably TIP3P)
characteristically *underestimate* viscosity, so a low value can reflect the
force field rather than the analysis. Always report the value with its
convergence status so the downstream validation panel can judge it against
measured data rather than trusting it blind.
