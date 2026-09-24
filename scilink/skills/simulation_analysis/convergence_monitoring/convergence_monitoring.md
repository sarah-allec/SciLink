---
description: Post-run observable convergence monitoring for MD production runs.
applies_to: molecular_dynamics
---

## Overview

After a successful MD production run that targets a time-averaged observable
(viscosity, diffusion coefficient, thermal conductivity, relaxation time),
call `check_observable_convergence` on the output directory to verify that
the computed properties have converged.

This does **not** apply to DFT, single-point, or geometry-optimization
calculations — only to MD production runs that produce time-series data
(trajectories, thermo logs).

## Interpretation

The tool returns a per-property convergence verdict based on the analysis
skill's own diagnostic flags (e.g. `plateau_reached` for Green-Kubo
viscosity, `extreme_narrowing` for T₁ relaxation). A property without a
convergence flag is assumed converged — only skills that explicitly report
convergence status are checked.

- **All converged** (`"converged": true`): report the results to the user.
- **One or more unconverged** (`"converged": false`): the computed values
  are not statistically reliable. Proceed to the extension protocol below.

## Implementation

When observables have not converged:

1. **Report** which properties have not converged, the convergence flag that
   triggered the detection, and the current (unreliable) value.

2. **Propose an extension** to the user. Two strategies are appropriate:

   - **Add independent replicas** (preferred for Green-Kubo transport
     properties): generate additional production runs with different velocity
     seeds. The analysis agent will automatically discover and combine all
     replica outputs, producing a block-averaged estimate with reduced
     statistical uncertainty.

   - **Extend the production run** (preferred when longer correlation times
     are needed): continue from the last restart file with more timesteps.

   Explain which strategy you recommend and why.

3. **Ask for user approval** before running additional simulations (unless
   operating in fully autonomous mode).

4. **Re-check convergence** after each extension by calling
   `check_observable_convergence` again on the updated output directory.

5. **Budget limit**: do not extend more than **3 times** without explicit
   user approval. After 3 extensions, report the best available result with
   a warning that convergence was not achieved.

## Validation

- Do not skip the convergence check and report a non-converged value as if
  it were reliable.
- Do not extend DFT or static calculations — convergence monitoring applies
  only to time-series observables.
- Do not treat the numeric value as the convergence signal — use the
  convergence flag. A value that looks reasonable can still be unconverged.
