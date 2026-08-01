---
description: HOMO-LUMO gap from a molecular quantum-chemistry calculation, read engine-agnostically with cclib.
technique: eigenvalue_gap
computes: [homo_lumo_gap]
requires: [molecular_qc_output]
---

## Overview

The HOMO-LUMO gap — the energy difference between the highest occupied and lowest
unoccupied molecular orbital — from a finite-molecule quantum-chemistry
calculation. It is read directly from the parsed molecular-orbital energies, so
the computation is trivial; the value is engine-agnostic because the output is
read with cclib, which parses NWChem, Gaussian, Psi4, ORCA and others uniformly.

## Implementation

1. **Read the output with cclib** — `import cclib; data = cclib.io.ccread(path)`.
   cclib auto-detects the program and format, so this works on whichever
   quantum-chemistry output is present without naming an engine. Use the
   `molecular_qc_output` file from DATA_FILES.

2. **Molecular-orbital energies:** `data.moenergies` is a list with one array of
   MO energies (in eV) per spin; `data.homos` is the array of HOMO indices per
   spin (0-based).

3. **Gap:** for each spin `s`, HOMO = `moenergies[s][homos[s]]`, LUMO =
   `moenergies[s][homos[s] + 1]`. For a closed-shell (restricted) calculation
   there is one spin. For an unrestricted calculation, take the overall HOMO as
   the highest occupied energy across spins and the overall LUMO as the lowest
   unoccupied across spins, and report `LUMO - HOMO`.

4. Report the gap in eV. Print one JSON object as the last stdout line:
   `{"status":"success","value":<gap in eV>,"units":"eV","homo_eV":<float>,
   "lumo_eV":<float>}`. On failure: `{"status":"error","message":<str>}`.

## Validation

- The gap must be **positive** (LUMO above HOMO); a negative gap indicates an
  orbital-ordering or spin-handling error.
- Typical closed-shell organic molecules have DFT gaps of ~2–8 eV (GGA
  functionals systematically *underestimate* gaps); a value far outside a few
  tenths to ~15 eV signals a unit or parsing error.
- If `moenergies`/`homos` are missing, the calculation did not converge or did
  not print orbital energies — report an error rather than a number.

## Interpretation

The HOMO-LUMO gap is a first-order indicator of chemical reactivity and optical
excitation energy. Because standard DFT underestimates gaps, compare like with
like (same functional/basis) and report the method — do not read an absolute
optical gap off a GGA number.
