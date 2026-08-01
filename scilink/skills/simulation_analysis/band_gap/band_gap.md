---
description: Electronic band gap from a periodic-DFT calculation, read engine-agnostically via pymatgen/ASE eigenvalues.
technique: eigenvalue_band_structure
computes: [band_gap]
requires: [dft_output]
---

## Overview

The fundamental electronic band gap of a periodic system — the energy difference
between the valence-band maximum (VBM) and conduction-band minimum (CBM) — read
from a DFT calculation's eigenvalues and occupations. The value comes straight
out of the parsed electronic structure; it is engine-agnostic because the output
is read with pymatgen/ASE, which parse VASP, Quantum ESPRESSO and others.

## Implementation

1. **Read the DFT output** from DATA_FILES using pymatgen or ASE — do not name an
   engine. For a VASP `vasprun.xml`, `pymatgen.io.vasp.Vasprun(path)` exposes
   `.eigenvalue_band_properties` → `(gap, cbm, vbm, is_direct)` directly. For
   other engines use the analogous pymatgen parser or read the eigenvalues via
   `ase.io.read` and the calculator's eigenvalue/occupation arrays.

2. **Compute the gap** if not given directly: from the eigenvalues ε(k, n) and
   their occupations, VBM = max occupied ε over all k and bands, CBM = min
   unoccupied ε; the fundamental gap is CBM − VBM. A negative (overlapping) gap
   means the system is metallic — report gap = 0.

3. **Direct vs indirect:** the gap is direct if the VBM and CBM occur at the same
   k-point. Report this when available.

4. Report the gap in eV. Print one JSON object as the last stdout line:
   `{"status":"success","value":<gap in eV>,"units":"eV","vbm_eV":<float>,
   "cbm_eV":<float>,"is_direct":<bool>}`. On failure: `{"status":"error",
   "message":<str>}`.

## Validation

- The gap must be **non-negative**; report 0 (metallic) when bands overlap rather
  than a negative number.
- Standard semilocal DFT (LDA/GGA) **systematically underestimates** band gaps
  (often by ~50%); a hybrid or GW result is larger. Judge against the same level
  of theory, and flag if the k-point sampling is too coarse to locate the true
  VBM/CBM (a common cause of a spuriously large or wrong gap).
- Typical semiconductor GGA gaps are ~0–3 eV; a value far outside a physically
  reasonable range signals a parsing or occupation error.

## Interpretation

The band gap sets whether the material is a metal, semiconductor, or insulator
and bounds its optical absorption onset. Because semilocal DFT underestimates
gaps, report the functional and do not compare a GGA gap directly to an
experimental optical gap.
